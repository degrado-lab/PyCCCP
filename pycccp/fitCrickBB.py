# Copyright (C) 2023 Rian Kormos
#   This program is free software: you can redistribute it and/or modify it 
#   under the terms of the GNU General Public License as published by the 
#   Free Software Foundation, either version 3 of the License, or (at your 
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but 
#   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
#   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License 
#   for more details.
#
#   You should have received a copy of the GNU General Public License along 
#   with this program. If not, see <http://www.gnu.org/licenses/>

import sys

import numpy as np
import numba as nb
from scipy.optimize import minimize

from Bio.PDB import PDBParser

from pycccp.generateCrickBB import *
from pycccp.PDBIO import *

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def fitCrickBB(pdbfile, cN, pType='GENERAL', IP=[], LB=[], UB=[], mask=[], 
               selection='', bbtype='ca', angle_units='degree'):
    """Fit Crick parameters to an input structure.

    Fits Crick parameters given the structure in pdbfile. If you use this 
    program in your research, please cite G. Grigoryan, W. F. DeGrado, 
    "Probing Designability via a Generalized Model of Helical Bundle 
    Geometry", J. Mol. Biol., 405(4): 1079-1100 (2011).

    Parameters
    ----------
    pdbfile : str
       Path to PDB file from which to extract CA atoms for fitting.
    cN : int
       The number of chains in the file. The number of coordinates has to be 
       divisible by cN.
    pType : str
       The parameterization type. All fits include superhelical radius (r0), 
       helical radius (r1), superhelical frequency (w0), helical frequency 
       (w1), and pitch angle (a) as global parameters. pType then controls 
       which of the remaining parameters are local to each chain and which 
       are global. Choices are:
           'GENERAL' -- this is the most general option and means that each 
           helix gets its own helical phase (ph1), superhelical phase offset 
           (dph0) and Z offset (Zoff), regardless of orientation
           'SYMMETRIC' -- the most symmetric option. ph1 is still individual 
           to each chain, but there is only one Zoff denoting the offset 
           between all parallel and antiparallel chains (i.e. all chains 
           sharing direction are assumed to have zero offset with respect to 
           each other). Also, for parallel topologies, dph0 is fixed at 
           i*2*pi. Nc for the i-th chain (so dph0 is not a parameter). For 
           strictly alternating topologies (up, down, up, down, ...), one 
           dph0 parameter is allowed to denote the relative rotation of all 
           parallel chains with respect to all anti-parallel chains. NOTE: 
           this requires that chains alternate direction in the order listed 
           in the input file.
           'ZOFF-SYMM' -- same as GENERAL, but Zoff is treated as in SYMMETRIC.
           'DPH0-SYMM' -- same as GENERAL, but dph0 is treated as in SYMMETRIC.
           'GENERAL-HLXPH' -- same as GENERAL, but all chains share ph1.
           'SYMMETRIC-HLXPH' -- same as SYMMETRIC, but all chains share ph1.
           'ZOFF-SYMM-HLXPH' -- same as ZOFF-SYMM, but all chains share ph1.
    IP : list
        Initial parameters for the search. Must be either empty (i.e. []), in 
        which case initialization is done automatically (recommended) or have 
        six entries, signifying starting values for the parameters r0, r1, w0, 
        w1, a, ph1 in that order (one value for each, even if various non-
        symmetric options are specified).
    LB : list
        Lower bounds for parameters. Can be left empty (recommended) or must 
        be of the same length as IP. If non-empty, UB must also be non-empty.
    UB : list
        Upper bounds for parameters. Can be left empty (recommended) or must 
        be of the same length as IP. If non-empty, LB must also be non-empty.
    mask : list
        If not empty, removes the contribution of certain atoms to the total 
        error (and hence the overall fit). Must have the same number of 
        elements as the total number of residues in the structure (after the 
        selection is applied; see below). Elements should be either 0 
        (indicating that the corresponding residue does not contribute) or 1 
        (indicating that the residue does contribute). The order of residues 
        starts with the first residue of the first chain and goes on to the 
        last residue of the last chain, as listed in pdbfile. So, for example, 
        if we have a trimer with 28 residues in each monomer, and we'd like to 
        remove the contribution of the C-terminal residues of the second chain, 
        mask would be set to [1]*50 + [0]*6 + [1]*28.
    selection : str
        If empty, the entire structure from pdbfile will be aligned against. 
        Otherwise, a selection of residues should be provided such that 
        discontiguous fragments are separated by colons and a one-letter chain 
        identifier is provided at the beginning of the each fragment that 
        begins a new chain. For example, "A7-32:39-64:B7-32:39-64" would 
        specify residues 7 to 32 (inclusive) and 39 to 64 for both chains A 
        and B of a structure.
    bbtype : str
        The type of backbone to generate for the fitted structure. It should 
        be one of the following strings:
            'ca' -- a backbone containing only alpha carbon (Ca) atoms
            'gly' -- a backbone containing N, Ca, C, O, and H atoms
            'ala' -- a backbone containing N, Ca, C, O, Cb, and H atoms
    angle_units : float
        Units of the angles that will be returned in params_dict.  Must be 
        either 'radian' or 'degree'.
    
    Outputs
    -------
    params_dict : dict
        Dict mapping the names to the values of the best-fit Crick parameters.
    err : float
        RMSD (in Angstroms) between ideal and input structure.
    xyz : np.array [n_atoms x 3]
        Array of coordinates of the ideal structure.
    chains : np.array [n_atoms]
        One-letter chain identifiers of each residue in the ideal structure.
    structure : BIO.PDB.Structure
        The structure from the PDB file, with coordinates transformed to align 
        with minimal SSD to the ideal structure.
    """
    if pType not in ['GENERAL', 'SYMMETRIC', 'ZOFF-SYMM', 'DPH0-SYMM', 
                     'GENERAL-HLXPH', 'SYMMETRIC-HLXPH', 'ZOFF-SYMM-HLXPH', 
                     'DPH0-SYMM-HLXPH']:
        raise ValueError('Unknown parameterization type {}'.format(pType))
    
    if len(IP) not in [0, 6]:
        raise ValueError('Unexpected number of initial parameters!')

    parser = PDBParser()
    structure = parser.get_structure('coil', pdbfile)
    if selection:
        M0 = select_coords(structure, selection)
    else:
        M0 = np.array([a.coord for a in structure.get_atoms() 
                       if a.name == 'CA'])

    if len(mask):
        assert len(mask) == len(M0)
        mask = np.array(mask)
        n = mask.sum()
    else:
        n = len(M0)
        mask = np.ones(n, dtype=bool)
    chL = n // cN
    if n % cN:
        raise ValueError(('Number of unmasked coordinates {} is not divisible '
                          'by the number of chains {}.').format(n, cN))
    
    # subtract centroid of input bundle coordinates and rotate such that the 
    # first principal axis of the input bundle aligns with Z and the centroid 
    # of the first chain has a Y-coordinate of 0
    M0bar = np.mean(M0[mask], axis=0)
    M = M0[mask] - M0bar
    H = np.dot(M.T, M)
    w, v = np.linalg.eigh(H)
    if np.dot(v[:, 2], M[chL] - M[0]) < 0:
        v *= -1. # ensure Z is approximately parallel to the first helix
    if np.linalg.det(v) < 0:
        v[:, 1] = -v[:, 1] # ensure the PC frame is right-handed
    M = np.dot(M, v) # rotate bundle into principal component (PC) frame
    structure.transform(v, -np.dot(v.T, M0bar))
    '''
    dist = np.zeros(cN) # centroid distances to the Z-axis
    centroid_0 = M[:chL].mean(axis=0)
    dist[0] = np.sqrt(centroid_0[0] ** 2 + centroid_0[1] ** 2)
    theta = (np.pi * (1. - np.sign(centroid_0[1])) + # compute angle required 
             np.sign(centroid_0[1]) *                # to rotate the bundle
             np.arccos(centroid_0[0] / dist[0]))     # about Z to put the  
                                                     # centroid of the first
                                                     # chain in the XZ plane 
    '''

    # compute rotation angle of each chain's centroid about the first PC axis, 
    # in the interval [0, 2*pi), along with orientations and crossing angles
    dist = np.zeros(cN) # centroid distances to the Z-axis
    chain_thetas = 2. * np.pi * np.ones(cN)
    cr = np.ones(cN) # orientations
    for i in range(cN):
        centroid = M[chL*i:chL*(i+1)].mean(axis=0)
        dist[i] = np.sqrt(centroid[0] ** 2 + centroid[1] ** 2)
        if i > 0:
            chain_thetas[i] = np.arctan2(centroid[1], centroid[0]) + np.pi
            cr[i] = (M[chL*(i+1)-1, 2] - M[chL*i, 2] > 0)
        else:
            assert (M[chL*(i+1)-1, 2] - M[chL*i, 2] > 0)
    '''
    for i in range(1, cN):
        centroid_i = M[chL*i:chL*(i+1)].mean(axis=0)
        dist[i] = np.sqrt(centroid_i[0] ** 2 + centroid_i[1] ** 2)
        theta = np.arctan2(centroid_i[1], centroid_i[0])
        # theta = np.pi * (1. - np.sign(centroid_i[1])) + \
        #         np.sign(centroid_i[1]) * \
        #         np.arccos(centroid_i[0] / dist[i])
        chain_thetas[i] = theta
        cr[i] = (M[chL*(i+1)-1, 2] - M[chL*i, 2] > 0)
    '''
    # determine chain order (clockwise)
    co = np.argsort(chain_thetas)[::-1]

    if len(IP) == 0:
        ideal = ideal_helix(7, start=-3)
        R, t, ssd, _ = kabsch(ideal, M[:7])
        sgn = np.sign(np.dot(t, np.cross(R[:, 2], np.array([0, 0, 1]))))
        alpha_guess = sgn * np.arccos(R[2, 2])
        IP = [np.mean(dist), # r0
              2.26, # r1
              1.51 * np.sin(alpha_guess) / np.mean(dist), # w0
              4. * np.pi / 7., # w1
              alpha_guess, # alpha
              np.arctan2(t[1], t[0])] # ph1
  
    p0 = prepare_guess(IP, cN, chL, co, cr, pType)
    res = minimize(crickSSD, p0, args=(M, pType, cN, co, cr, True),
                   method='L-BFGS-B', jac=True, tol=0.001)
    print(res.message)
    res.x = p0
    ssd, _, R, t = crickSSD(res.x, M, pType, cN, co, cr, False)
    err = np.sqrt(ssd / n)

    # align coordinates of initial structure with ideal structure
    structure.transform(R, t)

    r0, r1, w0, w1, a, ph1, zoff, dph0, J0 = \
        unpack_params(res.x, pType, cN, chL, co, cr)
    ph1 = fmod(ph1, 2. * np.pi)
    heptad = [getHeptadPos(ph1[i], False) for i in range(cN)]
    t_a = np.array([(7 - getHeptadPos(ph1[i], True)) % 7 for i in range(cN)])
    phCa = fmod(w1 * t_a + ph1 + np.pi, 2. * np.pi)
    XYZ, _ = crickBB(cN, chL, r0, r1, w0, w1, a, ph1, cr[1:], dph0, zoff)
    # determine pitch
    P = 2. * np.pi * r0 / np.abs(np.tan(a))
    # determine alternate definitions of zoff
    zoff_apNN = np.zeros_like(zoff)
    zoff_register = np.zeros_like(zoff)
    zoff_aa = np.zeros_like(zoff)
    for i in range(1, cN):
        if cr[i]:
            zo = 0.
        else:
            zo = XYZ[0, 2] - XYZ[(i+1)*chL-1, 2] + zoff[i - 1]
        dz_register = absoluteToRegisterZoff(zo, r0, w0, w1, a, 
                                             ph1[0], ph1[i], cr[i])
        dz_aa = absoluteToZoff_aa(zo, r0, r1, w0, w1, a, 
                                  ph1[0], ph1[i], cr[i])
        zoff_apNN[i - 1] = zoff[i - 1] - zo
        zoff_register[i - 1] = zoff[i - 1] - zo + dz_register
        zoff_aa[i - 1] = zoff[i - 1] - zo + dz_aa
    # adjust dph0 according to zoff, r0, and a to match the definition used 
    # by generateCrickBB.py
    dph0 = fmod(dph0 + zoff * np.tan(a) / r0, 2. * np.pi)
    
    xyz, chains = generateCrickBB(cN, chL, r0, r1, w0, w1, a, list(ph1), 
                                  list(cr[1:]), list(dph0), 
                                  list(zoff_register), None, 
                                  'registerzoff', 'ala', 'radian')
    
    if angle_units == 'degree':
        w0 *= 180. / np.pi
        w1 *= 180. / np.pi
        a *= 180. / np.pi
        ph1 *= 180. / np.pi
        phCa *= 180. / np.pi
        dph0 *= 180. / np.pi

    params_dict = {'cN' : cN, 'chL' : chL, 'r0' : r0, 'r1' : r1, 
                   'w0' : w0, 'w1' : w1, 'a' : a, 'pitch' : P, 
                   'cr' : list(cr[1:]), 'ph1' : list(ph1), 'phCa' : list(phCa),
                   'dph0' : list(dph0), 'zoff_apNN' : list(zoff_apNN), 
                   'zoff_register' : list(zoff_register), 
                   'zoff_aa' : list(zoff_aa), 
                   'starting_heptad_pos' : heptad}

    return params_dict, err, xyz, chains, structure
    

def prepare_guess(IP, cN, chL, co, cr, pType):
    """Prepare vector of initial parameters given a parameterization type.

    Parameters
    ----------
    IP : list
        Initial parameters for the search. Must have six entries, signifying 
        starting values for the parameters r0, r1, w0, w1, a, ph1.
    cN : int
        The number of chains in the ideal parametric backbone.
    chL : int
        The length of each chain (in amino acid residues).
    co : np.array [cN]
        The indices necessary to sort the chains into clockwise order, 
        beginning from the first chain.
    cr : np.array [cN - 1]
        For each chain k (k > 0), its orientation is parallel to the zeroth 
        chain if cr[k - 1] == 1 and anti-parallel to chain 0 if cr[k - 1] == 0.
    pType : str
       The parameterization type. All fits include superhelical radius (r0), 
       helical radius (r1), superhelical frequency (w0), helical frequency 
       (w1), and pitch angle (a) as global parameters. pType then controls 
       which of the remaining parameters are local to each chain and which 
       are global. Choices are:
           'GENERAL' -- this is the most general option and means that each 
           helix gets its own helical phase (ph1), superhelical phase offset 
           (dph0) and Z offset (Zoff), regardless of orientation
           'SYMMETRIC' -- the most symmetric option. ph1 is still individual 
           to each chain, but there is only one Zoff denoting the offset 
           between all parallel and antiparallel chains (i.e. all chains 
           sharing direction are assumed to have zero offset with respect to 
           each other). Also, for parallel topologies, dph0 is fixed at 
           i*2*pi.Nc for the i-th chain (so dph0 is not a parameter). For 
           strictly alternating topologies (up, down, up, down, ...), one 
           dph0 parameter is allowed to denote the relative rotation of all 
           parallel chains with respect to all anti-parallel chains. NOTE: 
           this requires that chains alternate direction in the order listed 
           in the input file.
           'ZOFF-SYMM' -- same as GENERAL, but Zoff is treated as in SYMMETRIC.
           'DPH0-SYMM' -- same as GENERAL, but dph0 is treated as in SYMMETRIC.
           'GENERAL-HLXPH' -- same as GENERAL, but all chains share ph1.
           'SYMMETRIC-HLXPH' -- same as SYMMETRIC, but all chains share ph1.
           'ZOFF-SYMM-HLXPH' -- same as ZOFF-SYMM, but all chains share ph1.

    Returns
    -------
    p0 : np.array
        Array of initial guess parameters for minimization.
    """
    p0 = [IP[i] for i in range(5)]
    chZ = IP[2] * chL * IP[0] / np.tan(IP[4]) # length of chain in Angstroms

    if 'HLXPH' in pType:
        p0 += [IP[5]]
    else:
        p0 += [IP[5]] * cN

    if 'GENERAL' in pType:
        p0 += [0. if cr[i] else chZ for i in range(1, cN)]
        p0 += [co[-1:0:-1][i] * 2. * np.pi / cN if cr[i + 1] else 
               co[-1:0:-1][i] * 2. * np.pi / cN + IP[2] * chL 
               for i in range(cN - 1)]
    
    if 'SYMMETRIC' in pType:
        p0 += [0.] # Only one Zoff parameter, no dph0

    if 'ZOFF-SYMM' in pType:
        p0 += [0.] # Only one Zoff parameter
        p0 += [co[-1:0:-1][i] * 2. * np.pi / cN for i in range(cN - 1)]

    if 'DPH0-SYMM' in pType:
        p0 += [0. if cr[i] else chZ for i in range(1, cN)]

    return np.array(p0)


def unpack_params(p, pType, cN, chL, co, cr):
    """Unpack a full set of Crick parameters from a reduced set corresponding 
       to a particular parameterization type.

    Parameters
    ----------
    p : np.array [n_param]
        Array of reduced parameters, with a length corresponding to the number 
        of free parameters given pType (see documentation for pType).
    pType : str
       The parameterization type. All fits include superhelical radius (r0), 
       helical radius (r1), superhelical frequency (w0), helical frequency 
       (w1), and pitch angle (a) as global parameters. pType then controls 
       which of the remaining parameters are local to each chain and which 
       are global. Choices are:
           'GENERAL' -- this is the most general option and means that each 
           helix gets its own helical phase (ph1), superhelical phase offset 
           (dph0) and Z offset (Zoff), regardless of orientation
           'SYMMETRIC' -- the most symmetric option. ph1 is still individual 
           to each chain, but there is only one Zoff denoting the offset 
           between all parallel and antiparallel chains (i.e. all chains 
           sharing direction are assumed to have zero offset with respect to 
           each other). Also, for parallel topologies, dph0 is fixed at 
           i*2*pi.Nc for the i-th chain (so dph0 is not a parameter). For 
           strictly alternating topologies (up, down, up, down, ...), one 
           dph0 parameter is allowed to denote the relative rotation of all 
           parallel chains with respect to all anti-parallel chains. NOTE: 
           this requires that chains alternate direction in the order listed 
           in the input file.
           'ZOFF-SYMM' -- same as GENERAL, but Zoff is treated as in SYMMETRIC.
           'DPH0-SYMM' -- same as GENERAL, but dph0 is treated as in SYMMETRIC.
           'GENERAL-HLXPH' -- same as GENERAL, but all chains share ph1.
           'SYMMETRIC-HLXPH' -- same as SYMMETRIC, but all chains share ph1.
           'ZOFF-SYMM-HLXPH' -- same as ZOFF-SYMM, but all chains share ph1. 
    cN : int
        The number of chains in the ideal parametric backbone.
    chL : int
        The length of each chain (in amino acid residues).
    co : np.array [cN]
        The indices necessary to sort the chains into clockwise order, 
        beginning from the first chain.
    cr : np.array [cN - 1]
        For each chain k (k > 0), its orientation is parallel to the zeroth 
        chain if cr[k - 1] == 1 and anti-parallel to chain 0 if cr[k - 1] == 0. 

    Returns
    -------
    r0 : float
        Superhelical radius (in Angstroms).
    r1 : float
        Helical radius (in Angstroms).
    w0 : float
        Superhelical frequency (in radians per residue).
    w1 : float
        Helical frequency (in radians per residue).
    a : float
        Pitch angle (in radians) of the superhelix.
    ph1 : np.array [cN]
        The helical phase angle (in radians) of each chain.
    dph0 : np.array [cN - 1]
        For each chain k (k > 0), the superhelical phase offset (in radians) 
        of chain k relative to chain 0 will be dph0[k - 1].
    zoff : np.array [cN - 1]
        For each chain k (k > 0), the Z-offset (in Angstroms) of chain k 
        relative to chain 0 will be zoff[k - 1].
    J0 : [5 + 3 * cN - 2 x n_param]
        Jacobian matrix of derivatives of full Crick parameter set with 
        respect to the reduced parameters.`
    """
    J0 = np.zeros((5 + 3 * cN - 2, len(p))) # Jacobian from reduced parameter 
                                            # space to full Crick param space

    r0, r1, w0, w1, a = p[:5]
    J0[:5, :5] = np.eye(5)
    if 'HLXPH' in pType:

        start, end = 5, 6
        ph1 = p[start] * np.ones(cN)
        J0[5:5+cN, start] = np.ones(cN)
    else:
        start, end = 5, 5 + cN
        ph1 = p[start:end]
        J0[5:5+cN, start:end] = np.eye(cN)

    if 'GENERAL' in pType:
        start, end = end, end + cN - 1
        zoff = p[start:end]
        J0[5+cN:5+2*cN-1, start:end] = np.eye(cN - 1)
        start, end = end, end + cN - 1
        dph0 = p[start:end]
        J0[5+2*cN-1:5+3*cN-2, start:end] = np.eye(cN - 1)

    if 'SYMMETRIC' in pType:
        zoff = np.zeros(cN)
        if 0 in cr:
            start, end = end, end + 1
            zoff[cr == 0] = p[start]
            J0[np.arange(5+cN,5+2*cN-1)[cr == 0], start] = \
                np.ones(np.sum(cr == 0))
        # check if the bundle has dihedral symmetry, i.e. alternating p and ap
        if np.all(cr[co[::2]] == 1) and np.all(cr[co[1::2]] == 0):
            start, end = end, end + 1
            dph0 = np.zeros(cN)
            si = np.argsort(np.argsort(co[cr == 1])) # clockwise order of 
                                                     # parallel chains
            dph0[cr == 1] = -si * 2. * np.pi / np.sum(cr == 1)
            si = np.argsort(np.argsort(co[cr == 0])) # clockwise order of 
                                                     # anti-parallel chains
            dph0[cr == 0] = p[start] - si * 2. * np.pi / np.sum(cr == 0)
            J0[np.arange(5+2*cN-1,5+3*cN-2)[cr == 0], start] = \
                np.ones(np.sum(cr == 0))
        else: # the bundle has Cn symmetry
            dph0 = -co * 2. * np.pi / cN

    if 'ZOFF-SYMM' in pType:
        zoff = np.zeros(cN)
        if 0 in cr:
            start, end = end, end + 1
            zoff[cr == 0] = p[start]
            J0[np.arange(5+cN,5+2*cN-1)[cr == 0], start] = \
                np.ones(np.sum(cr == 0))
        start, end = end, end + cN - 1
        dph0 = p[start:end]
        J0[5+2*cN-1:5+3*cN-2, start:end] = np.ones(cN - 1)

    if 'DPH0-SYMM' in pType:
        start, end = end, end + cN - 1
        zoff = p[start:end]
        J0[5+cN:5+2*cN-1, start:end] = np.ones(cN - 1)
        dph0 = -co * 2. * np.pi / cN

    a = np.abs(a) * np.sign(w0) # make sure pitch angle and frequency 
                                # have the same sign
    
    return r0, r1, w0, w1, a, ph1, zoff, dph0, J0


def crickSSD(p, M, pType, cN, co, cr, is_fit=True):
    """Returns the error and its gradient between the coordinate set and the 
       ideal backbone given Crick parameters.

    Parameters
    ----------
    p : np.array [n_params]
        Array of parameter values for which to compute the error.
    M : np.array [n_atoms x 3]
        Coordinate set relative to which to compute the error of the ideal 
        parametric backbone.
    pType : str
       The parameterization type. All fits include superhelical radius (r0), 
       helical radius (r1), superhelical frequency (w0), helical frequency 
       (w1), and pitch angle (a) as global parameters. pType then controls 
       which of the remaining parameters are local to each chain and which 
       are global. Choices are:
           'GENERAL' -- this is the most general option and means that each 
           helix gets its own helical phase (ph1), superhelical phase offset 
           (dph0) and Z offset (Zoff), regardless of orientation
           'SYMMETRIC' -- the most symmetric option. ph1 is still individual 
           to each chain, but there is only one Zoff denoting the offset 
           between all parallel and antiparallel chains (i.e. all chains 
           sharing direction are assumed to have zero offset with respect to 
           each other). Also, for parallel topologies, dph0 is fixed at 
           i*2*pi.Nc for the i-th chain (so dph0 is not a parameter). For 
           strictly alternating topologies (up, down, up, down, ...), one 
           dph0 parameter is allowed to denote the relative rotation of all 
           parallel chains with respect to all anti-parallel chains. NOTE: 
           this requires that chains alternate direction in the order listed 
           in the input file.
           'ZOFF-SYMM' -- same as GENERAL, but Zoff is treated as in SYMMETRIC.
           'DPH0-SYMM' -- same as GENERAL, but dph0 is treated as in SYMMETRIC.
           'GENERAL-HLXPH' -- same as GENERAL, but all chains share ph1.
           'SYMMETRIC-HLXPH' -- same as SYMMETRIC, but all chains share ph1.
           'ZOFF-SYMM-HLXPH' -- same as ZOFF-SYMM, but all chains share ph1. 
    cN : int
        The number of chains in the ideal parametric backbone.
    co : np.array [cN]
        The indices necessary to sort the chains into clockwise order, 
        beginning from the first chain.
    cr : np.array [cN - 1]
        For each chain k (k > 0), its orientation is parallel to the zeroth 
        chain if cr[k - 1] == 1 and anti-parallel to chain 0 if cr[k - 1] == 0.
    is_fit : bool
        If True, this function is being used within a nonlinear least-squares 
        fitting routine.

    Returns
    -------
    ssd : float
        Sum squared deviation (error) between the coordinate set the the 
        ideal backbone given Crick parameters.
    J : np.array [n_params]
        Gradient of error function with respect to the parameters.
    R : np.array [3 x 3] (only if is_fit is False)
        Rotation matrix required to align M with the ideal coordinates.
    t : np.array [3] (only if is_fit is False)
        Translation vector required to align M with the ideal coordinates.
    """
    n = len(M)
    chL = n // cN
    r0, r1, w0, w1, a, ph1, zoff, dph0, J0 = \
        unpack_params(p, pType, cN, chL, co, cr)
    cr = cr[1:] # switch to cr with length cN - 1

    XYZ, J_Crick = crickBB(cN, chL, r0, r1, w0, w1, a, ph1, cr, dph0, zoff)
    R, t, ssd, d_ssd_dXYZ = kabsch(M, XYZ)

    J = np.dot(np.sum(d_ssd_dXYZ * J_Crick, axis=(1, 2)), J0)
    if is_fit:
        return ssd, J
    else:
        return ssd, J, R, t


def crickBB(cN, chL, r0, r1, w0, w1, a, ph1, cr, dph0, zoff):
    """Compute the coordinates of a Crick-parameterized helical bundle and 
       their derivatives with respect to the real-valued parameters.

    Parameters
    ----------
    cN : int
        Number of chains.
    chL : int
        The length of each chain (in amino acid residues).
    r0 : float
        Superhelical radius (in Angstroms).
    r1 : float
        Helical radius (in Angstroms).
    w0 : float
        Superhelical frequency (in radians per residue).
    w1 : float
        Helical frequency (in radians per residue).
    a : float
        Pitch angle (in radians) of the superhelix.
    ph1 : np.array [cN]
        The helical phase angle (in radians) of each chain.
    cr : np.array [cN - 1]
        For each chain k (k > 0), its orientation is parallel to the zeroth 
        chain if cr[k - 1] == 1 and anti-parallel to chain 0 if cr[k - 1] == 0.
    dph0 : np.array [cN - 1]
        For each chain k (k > 0), the superhelical phase offset (in radians) 
        of chain k relative to chain 0 will be dph0[k - 1].
    zoff : np.array [cN - 1]
        For each chain k (k > 0), the Z-offset (in Angstroms) of chain k 
        relative to chain 0 will be zoff[k - 1].
    
    Returns
    -------
    xyz : np.array [cN * chL x 3]
        Coordinates of the Crick-parameterized helical bundle.
    J_Crick : np.array [5 + 3 * cN - 2 x cN * chL x 3]
        Jacobian of the coordinates of the Crick-parameterized helical 
        bundle with respect to the Crick parameters in the following 
        order: r0, r1, w0, w1, a, ph1 (cN values), zoff (cN - 1 values), 
        dph0 (cN - 1 values).
    """
    t = np.hstack([np.arange(chL)] * cN)
    ph1_orig, zoff_orig = ph1, zoff
    ph1, ph0, zoff = np.zeros(cN * chL), np.zeros(cN * chL), np.zeros(cN * chL)
    ph1[:chL] = ph1_orig[0]
    signs = np.ones(cN * chL)
    masks = np.zeros((cN, cN * chL))
    masks[0, :chL] = 1.
    for i in range(1, cN):
        ph1[i*chL:(i+1)*chL] = ph1_orig[i]
        ph0[i*chL:(i+1)*chL] = dph0[i - 1]
        zoff[i*chL:(i+1)*chL] = zoff_orig[i - 1]
        signs[i*chL:(i+1)*chL] = -1. + 2. * cr[i - 1]
        masks[i, i*chL:(i+1)*chL] = 1.
    w0 *= signs
    w1 *= signs
    ph1 *= signs
    cos_0, sin_0, cos_1, sin_1 = np.cos(w0 * t + ph0), np.sin(w0 * t + ph0), \
                                 np.cos(w1 * t + ph1), np.sin(w1 * t + ph1)
    cos_a, sin_a, tan_a = np.cos(a), np.sin(a), np.tan(a)
    xyz = np.array([r0 * cos_0 + r1 * cos_0 * cos_1 - 
                    r1 * cos_a * sin_0 * sin_1, 
                    r0 * sin_0 + r1 * sin_0 * cos_1 + 
                    r1 * cos_a * cos_0 * sin_1, 
                    r0 * w0 * t / tan_a - 
                    r1 * sin_a * sin_1 + zoff]).T
    J_x = np.vstack([cos_0, 
                     cos_0 * cos_1 - cos_a * sin_0 * sin_1, 
                     -t * (r0 * sin_0 + r1 * sin_0 * cos_1 + 
                           r1 * cos_a * cos_0 * sin_1) * signs, 
                     -t * r1 * (cos_0 * sin_1 + cos_a * sin_0 * cos_1) * signs, 
                     r1 * sin_a * sin_0 * sin_1, 
                     -r1 * (cos_0 * sin_1 + cos_a * sin_0 * cos_1) * 
                     masks * signs, 
                     np.zeros((cN - 1, cN * chL)), 
                     (-r0 * sin_0 - r1 * sin_0 * cos_1 - 
                      r1 * cos_a * cos_0 * sin_1) * masks[1:]])
    J_y = np.vstack([sin_0, 
                     sin_0 * cos_1 + cos_a * cos_0 * sin_1, 
                     t * (r0 * cos_0 + r1 * cos_0 * cos_1 - 
                          r1 * cos_a * sin_0 * sin_1) * signs, 
                     -t * r1 * (sin_0 * sin_1 - cos_a * cos_0 * cos_1) * signs, 
                     -r1 * sin_a * cos_0 * sin_1, 
                     -r1 * (sin_0 * sin_1 - cos_a * cos_0 * cos_1) * 
                     masks * signs, 
                     np.zeros((cN - 1, cN * chL)), 
                     (r0 * cos_0 + r1 * cos_0 * cos_1 - 
                      r1 * cos_a * sin_0 * sin_1) * masks[1:]])
    J_z = np.vstack([w0 * t / tan_a, 
                     -sin_a * sin_1, 
                     r0 * t / tan_a * signs, 
                     -r1 * t * sin_a * cos_1 * signs, 
                     -r0 * w0 * t / sin_a ** 2 - r1 * cos_a * sin_1, 
                     -r1 * sin_a * cos_1 * masks * signs, 
                     masks[1:], 
                     np.zeros((cN - 1, cN * chL))])
    J_x = J_x.reshape((J_x.shape[0], J_x.shape[1], 1))
    J_y = J_y.reshape((J_y.shape[0], J_y.shape[1], 1))
    J_z = J_z.reshape((J_z.shape[0], J_z.shape[1], 1))
    J_Crick = np.concatenate([J_x, J_y, J_z], axis=2)
    return xyz, J_Crick
    
        
def crossingAngle(A, B, pap):
    """Compute crossing angle between two helices with coordinates A and B.

    Parameters
    ----------
    A : np.array [M x 3]
        Coordinates of alpha carbon atoms in the first helix.
    B : np.array [N x 3]
        Coordinates of alpha carbon atoms in the second helix.
    pap : int
        Integer flag determining whether the helices are parallel (1) or 
        antiparallel (0).

    Returns
    -------
    a : float
        Crossing angle between the two helices.
    """
    if A.shape[1] != 3 or B.shape[1] != 3:
        raise ValueError(('Unexpected matrix size: A is [{} x {}], '
                          'B is [{} x {}]!'.format(A.shape[0], 
                                                   A.shape[1], 
                                                   B.shape[0], 
                                                   B.shape[1])))
    if pap == 0:
        B = B[::-1]

    if np.min([A.shape[0], B.shape[0]]) < 3:
        return 0.

    # find helical axes
    axsA = helicalAxisPoints(A)
    axsB = helicalAxisPoints(B)

    return dihe(axsA[0], axsA[-1], axsB[-1], axsB[0])


def helicalAxisPoints(H):
    """Determine the points along a helical axis given helix coordinates H.

    Parameters
    ----------
    H : np.array [N x 3]
        Coordinates of alpha carbon atoms in the helix.

    Returns
    -------
    axs : np.array [N - 2 x 3]
        Coordinates of points along the helical axis.
    """
    if H.shape[0] < 3:
        raise ValueError('Insufficient points to determine the helical axis.')

    r = (H[:-2] - H[1:-1]) + (H[2:] - H[1:-1])
    return 2.26 * r / np.linalg.norm(r, axis=1, keepdims=True) + H[1:-1]


def dihe(p1, p2, p3, p4):
    """Calculate dihedral angle formed by four points.

    Parameters
    ----------
    p1 : np.array [3]
        Coordinates of first point.
    p2 : np.array [3]
        Coordinates of second point.
    p3 : np.array [3]
        Coordinates of third point.
    p4 : np.array [3]
        Coordinates of fourth point.

    Returns
    -------
    d : float
        The dihedral angle formed by the four points.
    """
    v21 = p2 - p1
    v32 = p3 - p2
    v43 = p4 - p3

    px1 = np.cross(v21, v32)
    px1 /= np.linalg.norm(px1)

    px2 = np.cross(v32, v43)
    px2 /= np.linalg.norm(px2)

    x = np.dot(px1, px2)
    y = np.dot(np.cross(px1, v32 / np.linalg.norm(v32)), px2) 

    return np.arctan2(y, x)


def ideal_helix(n_residues, rise=1.5, twist=100, radius=2.3, start=0):
    """Generate coordinates for an ideal helix.

    Parameters
    ----------
    n_residues : int
        Number of residues in the helix.
    rise : float, optional
        Rise per residue of the helix in Angstroms. Default is 1.5.
    twist : float, optional
        Twist per residue of the helix in degrees. Default is 100.
    radius : float, optional
        Radius of the helix in Angstroms. Default is 2.3.
    start : int, optional
        Which residue to start the helix on. The residue with index
        0 will always lie along the x-axis.  Default is 0.

    Returns
    -------
    coords : np.array [n_points x 3]
        Array of coordinates for the ideal helix.
    """
    t = np.linspace(start * twist * np.pi / 180,
                    (n_residues + start) * twist * np.pi / 180,
                    n_residues, endpoint=False)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(0, n_residues * rise, n_residues,
                    endpoint=False)
    return np.array([x, y, z]).T


def kabsch(X, Y):
    """Rotate and translate X into Y to minimize the SSD between the two, 
       and find the derivatives of the SSD with respect to the entries of Y. 
       
       Implements the SVD method by Kabsch et al. (Acta Crystallogr. 1976, 
       A32, 922) and the SVD differentiation method by Papadopoulo and 
       Lourakis (INRIA Sophia Antipolis. 2000, research report no. 3961).

    Parameters
    ----------
    X : np.array [N x 3]
        Array of mobile coordinates to be transformed by a proper rotation 
        to minimize sum squared displacement (SSD) from Y.
    Y : np.array [N x 3]
        Array of stationary coordinates against which to transform X.

    Returns
    -------
    R : np.array [3 x 3]
        Proper rotation matrix required to transform X such that its SSD 
        with Y is minimized.
    t : np.array [3]
        Translation matrix required to transform X such that its SSD with Y 
        is minimized.
    ssd : float
        Sum squared displacement after alignment.
    d_ssd_dY : np.array [N x 3]
        Matrix of derivatives of the SSD with respect to each element of Y.
    """
    n = len(X)
    # compute R using the Kabsch algorithm
    Xbar, Ybar = np.mean(X, axis=0), np.mean(Y, axis=0)
    Xc, Yc = X - Xbar, Y - Ybar
    H = np.dot(Xc.T, Yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.dot(U, Vt)))
    D = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., d]])
    R = np.dot(U, np.dot(D, Vt))
    t = Ybar - np.dot(Xbar, R)
    # compute SSD from aligned coordinates XR
    XRmY = np.dot(Xc, R) - Yc
    ssd = np.sum(XRmY ** 2)
    # compute derivative of R with respect to Y
    omega_U, omega_Vt = populate_omegas(U, S, Vt)
    dUdH = np.einsum('km,ijml->ijkl', U, omega_U)
    dVtdH = -np.einsum('ijkm,ml->ijkl', omega_Vt, Vt)
    dRdH = np.einsum('imkl,mj->ijkl', dUdH, np.dot(D, Vt)) + \
           np.einsum('im,mjkl->ijkl', np.dot(U, D), dVtdH)
    dRdY = np.einsum('km,ijml->ijkl', Xc, dRdH)
    XdRdY = np.einsum('im,mjkl->ijkl', Xc, dRdY)
    d_ssd_dY = 2. * (np.sum(XRmY * XdRdY, axis=(0, 1)) - XRmY) 
    return R, t, ssd, d_ssd_dY


@nb.jit(nopython=True, cache=True)
def populate_omegas(U, S, Vt):
    """Populate omega_U and omega_Vt matrices from U, S, and Vt.

    Parameters
    ----------
    U : np.array [3 x 3]
        Left unitary matrix from a singular value decomposition.
    S : np.array [3]
        Vector of singular values from a singular value decomposition.
    Vt : np.array [3 x 3]
        Right unitary matrix from a singular value decomposition.

    Returns
    -------
    omega_U : np.array [3 x 3 x 3 x 3]
        omega_U matrix of matrices as described in Papadopolou and 
        Lourakis (2000).
    omega_Vt : np.array [3 x 3 x 3 x 3]
        omega_V matrix of matrices as described in Papadopolou and 
        Lourakis (2000), but with the last two dimensions transposed.
    """
    omega_U = np.zeros((3, 3, 3, 3))
    omega_Vt = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k, l in [(0, 1), (1, 2), (2, 0)]:
                system_A = np.array([[S[l], S[k]], 
                                     [S[k], S[l]]])
                system_b = np.array([[U[i, k] * Vt[l, j]], 
                                     [-U[i, l] * Vt[k, j]]])
                if S[k] != S[l]:
                    soln = np.linalg.solve(system_A, system_b)
                else: # solve via least squares in the degenerate case
                    soln, _, _, _ = np.linalg.lstsq(system_A, system_b, 1e-14)
                omega_U[i, j, k, l], omega_Vt[i, j, l, k] = soln.flatten()
                omega_U[i, j, l, k] = -omega_U[i, j, k, l]
                omega_Vt[i, j, k, l] = -omega_Vt[i, j, l, k]
    return omega_U, omega_Vt


def select_coords(structure, selection):
    """Select CA coordinates from a structure given a selection string.

    Parameters
    ----------
    structure : Bio.PDB.Structure
        The structure from the PDB file, upon which to perform the selection.
    selection : str 
        If empty, the entire structure from pdbfile will be aligned against. 
        Otherwise, a selection of residues should be provided such that 
        discontiguous fragments are separated by colons and a one-letter chain 
        identifier is provided at the beginning of the each fragment that 
        begins a new chain. For example, "A7-32:39-64:B7-32:39-64" would 
        specify residues 7 to 32 (inclusive) and 39 to 64 for both chains A 
        and B of a structure.
    
    Returns
    -------
    ca_coords : np.array [n_atoms x 3]
        The coordinates of the CA atoms from the selected residues.
    """
    fragments = selection.split(':')
    if fragments[0][0] not in alphabet:
        raise ValueError(('The first fragment in the selection must have '
                          'a chain ID!'))
    sel_chains_resnums = [] 
    current_chain = fragments[0][0]
    for fragment in fragments:
        if fragment[0] == current_chain:
            fragment = fragment[1:]
        elif fragment[0] in alphabet:
            current_chain = fragment[0]
            fragment = fragment[1:]
        if '-' in fragment:
            start, end = [int(val) for val in fragment.split('-')]
        else:
            start, end = int(fragment), int(fragment)
        for i in range(start, end + 1):
            sel_chains_resnums.append((current_chain, i))
    chains_resnums_to_coords = {}
    for a in structure.get_atoms():
        chain_resnum = (a.get_parent().get_parent().id, 
                        a.get_parent().id[1])
        if chain_resnum not in chains_resnums_to_coords.keys() \
                and a.name == 'CA':
            chains_resnums_to_coords[chain_resnum] = a.coord
    return np.array([chains_resnums_to_coords[chain_resnum] 
                     for chain_resnum in sel_chains_resnums])
