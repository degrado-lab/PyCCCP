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

import inspect
import numpy as np

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def generateCrickBB(cN, chL, r0, r1, w0, w1, a, ph1, cr, dph0, zoff, 
                    dt=None, ztype='', bbtype='ca', angle_unit='degree'):
    """Generate a coiled-coil backbone from Crick parameters.

    Generates an ideal coiled-coil backbone from given Crick parameters. If 
    you use this program in your research, please cite G. Grigoryan, W. F. 
    DeGrado, "Probing Designability via a Generalized Model of Helical Bundle 
    Geometry," J. Mol. Biol., 405(4): 1079-1100 (2011).

    Parameters
    ----------
    cN : int
        Number of chains.
    chL : int or list [cN]
        If an int, the length of all chains (in amino acid residues). If a 
        list of int, the length of each chain (in amino acid residues).
    r0 : float
        Superhelical radius (in Angstroms).
    r1 : float
        Helical radius (in Angstroms).
    w0 : float
        Superhelical frequency (in degrees per residue).
    w1 : float
        Helical frequency (in degrees per residue).
    a : float
        Pitch angle (in degrees) of the superhelix.
    ph1 : float or list [cN]
        If a float, the helical phase angle (in degrees) of all chains. If a 
        list of float, the helical phase angle (in degrees) of each chain.
    cr : list [cN - 1]
        For each chain k (k > 0), its orientation is parallel to the zeroth 
        chain if cr[k - 1] == 1 and anti-parallel to chain 0 if 
        cr[k - 1] == 0.
    dph0 : list [cN - 1]
        For each chain k (k > 0), the superhelical phase offset (in degrees) 
        of chain k relative to chain 0 will be dph0[k - 1].
    zoff : list [cN - 1]
        For each chain k (k > 0), the Z-offset (in Angstroms) of chain k 
        relative to chain 0 will be zoff[k - 1].
    dt : list [cN]
        Number of additional residues to append to the termini of each chain.
    ztype : str
        The type of Z offset to use. It should be one of the following strings
        (see Grigoryan, DeGrado, JMB 405:1079, 2011 for exact definitions and 
        details):
            'zoffaa' -- the Z offset is interpreted as the offset between 'a' 
            positions on opposing chains.
            'registerzoff' -- the Z offset refers to the offset between points
            on the Crick curves of opposing chains that point directly into 
            the interface.
            'apNNzoff' -- the Z offset is interpreted as being between the N 
            termini of chains 0 and k if k runs parallel to 0, and the N 
            terminus of chain 0 and the C terminus of chain k if k is anti-
            parallel to chain 0.
    bbtype : str
        The type of backbone to generate. It should be one of the following 
        strings:
            'ca' -- a backbone containing only alpha carbon (Ca) atoms
            'gly' -- a backbone containing N, Ca, C, O, and H atoms
            'ala' -- a backbone containing N, Ca, C, O, Cb, and H atoms
    angle_unit : str
        The unit of angles passed as arguments to this function, either 
        'degree' or 'radian'.

    Returns
    -------
    XYZ : np.array [N_atoms x 3]
        Coordinates (in Angstroms) of backbone atoms.
    chains : np.array [N_atoms]
        Single-letter chain IDs of backbone atoms.
    """
    # convert optional int- or float- valued arguments to lists
    if type(chL) is not list:
        chL = [chL] * cN
    if type(ph1) is not list:
        ph1 = [ph1] * cN
    # convert lists to numpy arrays
    chL, ph1, cr, dph0, zoff, dt = np.array(chL), np.array(ph1), \
                                   np.array(cr), np.array(dph0), \
                                   np.array(zoff), np.array(dt)
    if angle_unit == 'degree':
        # convert degrees to radians
        w0 *= np.pi / 180.
        w1 *= np.pi / 180.
        a *= np.pi / 180.
        ph1 *= np.pi / 180.
        dph0 *= np.pi / 180.
    # prepend chain 0 values to arguments with length cN - 1
    cr = np.insert(cr, 0, 1)
    dph0 = np.insert(dph0, 0, 0.)
    zoff = np.insert(zoff, 0, 0.)
    # handle dt if it has a value of None
    if dt == None:
        dt = np.zeros(cN, dtype=int)
    # ensure all arrays have length cN
    incorrect = []
    for arg in [chL, ph1, cr, dph0, zoff, dt]:
        if len(arg) != cN:
            incorrect.append(retrieve_name(arg))
    if len(incorrect) == 1: 
        raise ValueError(('Argument {} was provided incorrectly. '
                          'Please refer to the documentation for correct '
                          'usage of generateCrickBB.').format(incorrect[0]))
    if len(incorrect) > 1:
        incorrect = ', '.join(incorrect[:-1]) + ' and ' + incorrect[-1]
        raise ValueError(('Arguments {} were provided incorrectly. '
                          'Please refer to the documentation for correct '
                          'usage of generateCrickBB.').format(incorrect))

    if bbtype == 'ca':
        res_natoms = 1
    elif bbtype == 'gly':
        res_natoms = 5
    elif bbtype == 'ala':
        res_natoms = 6

    chL_eff = chL + 2 * dt
    cumul_nres = [np.sum(chL_eff[:i], dtype=int) for i in range(cN + 1)]
    XYZ = np.zeros((res_natoms * cumul_nres[-1], 3))
    chains = np.empty(res_natoms * cumul_nres[-1], dtype=object)
    t = [np.arange(-dt[i], chL[i] + dt[i]).astype(int) for i in range(cN)]
    for i in range(cN):
        if cr[i]: # chain is parallel to chain 0
            xyz = crickEQ(r0, r1, w0, w1, a, 0, ph1[i], t[i], bbtype)
            if ztype == 'registerzoff':
                zo = 0 # Start with z-offset that brings termini together. 
                       # What register z-offset does this give?
                dz = absoluteToRegisterZoff(zo, r0, w0, w1, a, 
                                            ph1[0], ph1[i], cr[i])
                # Correct initial z-offset to give one with a 0 register
                # z-offset. This will be the basal z-offset, to which the 
                # specified offset will be added.
                zo = zo - dz
                zoff[i] = zoff[i] + zo
            elif ztype == 'zoffaa':
                zo = 0 # Start with z-offset that brings termini together. 
                       # What Zaa' does this give?
                dz = absoluteToZoff_aa(zo, r0, r1, w0, w1, a, 
                                       ph1[0], ph1[i], cr[i])
                # Correct initial z-offset to give one with a 0 Z_aa' 
                # offset. This will be the basal z-offset, to which the 
                # specified offset will be added.
                zo = zo - dz
                zoff[i] = zoff[i] + zo 
        else: # chain is antiparallel to chain 0
            xyz = crickEQ(r0, r1, -w0, -w1, a, 0, -ph1[i], t[i], bbtype)
            if ztype == 'apNNzoff':
                zoff[i] = zoff[i] + XYZ[res_natoms*cumul_nres[i], 2] - \
                          xyz[-res_natoms*(1 + dt[i]), 2]
            elif ztype == 'registerzoff':
                if res_natoms == 1:
                    zo = XYZ[0, 2] - xyz[-1, 2]
                else:
                    zo = XYZ[1, 2] - xyz[-res_natoms+1, 2]
                # Start with z-offset that brings termini together. 
                # What register z-offset does this give?
                dz = absoluteToRegisterZoff(zo, r0, w0, w1, a, 
                                            ph1[0], ph1[i], cr[i])
                # Correct initial z-offset to give one with a 0 register
                # z-offset. This will be the basal z-offset, to which the 
                # specified offset will be added.
                zo = zo - dz
                zoff[i] = zoff[i] + zo
            elif ztype == 'zoffaa':
                if res_natoms == 1:
                    zo = XYZ[0, 2] - xyz[-1, 2]
                else:
                    zo = XYZ[1, 2] - xyz[-res_natoms+1, 2]
                # Start with z-offset that brings termini together. 
                # What Zaa' does this give?
                dz = absoluteToZoff_aa(zo, r0, r1, w0, w1, a, 
                                       ph1[0], ph1[i], cr[i])
                # Correct initial z-offset to give one with a 0 Z_aa' 
                # offset. This will be the basal z-offset, to which the 
                # specified offset will be added.
                zo = zo - dz
                zoff[i] = zoff[i] + zo
        T = np.array([[np.cos(dph0[i] - zoff[i] * np.tan(a) / r0), 
                       np.sin(dph0[i] - zoff[i] * np.tan(a) / r0), 0.], 
                      [-np.sin(dph0[i] - zoff[i] * np.tan(a) / r0), 
                       np.cos(dph0[i] - zoff[i] * np.tan(a) / r0), 0.], 
                      [0., 0., 1.]])
        xyz = np.dot(xyz, T)
        XYZ[res_natoms*cumul_nres[i]:
            res_natoms*cumul_nres[i + 1]] = xyz
        XYZ[res_natoms*cumul_nres[i]:
            res_natoms*cumul_nres[i + 1], 2] += zoff[i]
        chains[res_natoms*cumul_nres[i]:
               res_natoms*cumul_nres[i + 1]] = alphabet[i]
    return XYZ, chains


def absoluteToRegisterZoff(zoff, r0, w0, w1, a, ph1_1, ph1_2, p_ap):
    """Convert from an absolute Z offset to a register Z offset (defined as 
       the Z distance between points most inner to the bundle - i.e. points 
       with phase pi).

    Parameters
    ---------- 
    zoff : float
        Absolute Z-offset (in Angstroms) of the second chain relative to the 
        first chain.
    r0 : float
        Superhelical radius (in Angstroms).
    w0 : float
        Superhelical frequency (in radians per residue).
    w1 : float
        Helical frequency (in radians per residue).
    a : float
        Pitch angle (in radians).
    ph1_1 : float
        Helical phase angle (in radians) of the first chain.
    ph1_2 : float
        Helical phase angle (in radians) of the second chain.
    p_ap : int
        If 1, the chains are parallel. If 0, the chains are antiparallel.

    Returns
    -------
    rzoff : float
        Register Z offset (in Angstroms) of the second chain relative to the 
        first chain.
    """
    assert p_ap in [0, 1] # p_ap flag is expected to be either 0 or 1

    aa1 = 2. * np.pi / w1
    b1 = (np.pi - ph1_1) / w1
    z1 = (r0 * w0 / np.tan(a)) * (aa1 + b1)
    if p_ap: # chains are parallel
        b2 = (np.pi - ph1_2) / w1
        n = ((z1 - zoff) / (r0 * w0 / np.tan(a)) - b2) / aa1
        dz = (r0 * w0 / np.tan(a)) * (aa1 * np.floor(n) + b2) + zoff - z1
        dz1 = (r0 * w0 / np.tan(a)) * (aa1 * np.ceil(n) + b2) + zoff - z1
    else: # chains are antiparallel
        # for a chain running in the opposite orientation, the meaning of 
        # clockwise sense changes, so w1, w1 and phase flip sign
        aa2 = 2. * np.pi / (-w1)
        b2 = (np.pi + ph1_2) / (-w1)
        n = ((z1 - zoff) / (-r0 * w0 / np.tan(a)) - b2) / aa2
        dz = -(r0 * w0 / np.tan(a)) * (aa2 * np.floor(n) + b2) + zoff - z1
        dz1 = -(r0 * w0 / np.tan(a)) * (aa2 * np.ceil(n) + b2) + zoff - z1

    if np.abs(dz1) < np.abs(dz):
        return dz1
    else:
        return dz


def absoluteToZoff_aa(zoff, r0, r1, w0, w1, a, ph1_1, ph1_2, p_ap):
    """Convert from absolute Z offset to Zaa' Z offset, defined as the Z 
       distance between closest 'a' positions on opposite chains.

    Parameters
    ---------- 
    zoff : float
        Absolute Z-offset (in Angstroms) of the second chain relative to the 
        first chain.
    r0 : float
        Superhelical radius (in Angstroms).
    r1 : float
        Helical radius (in Angstroms).
    w0 : float
        Superhelical frequency (in radians per residue).
    w1 : float
        Helical frequency (in radians per residue).
    a : float
        Pitch angle (in radians).
    ph1_1 : float
        Helical phase angle (in radians) of the first chain.
    ph1_2 : float
        Helical phase angle (in radians) of the second chain.
    p_ap : int
        If 1, the chains are parallel. If 0, the chains are antiparallel.

    Returns
    -------
    zaa : float
        Zaa' Z offset (in Angstroms) of the second chain relative to the 
        first chain.
    """
    assert p_ap in [0, 1] # p_ap flag is expected to be either 0 or 1

    # find first a-positions on chain 1
    rng = np.arange(0, 7)
    mi = np.argmin(np.abs(                      # the position closest to the 
        angleDiff(ph1_1 + w1 * rng,             # canonical 'a' position 
                  canonicalPhases(0, 'radian')) # phase in the first heptad 
        ))                                      # plus a bit
    aph1_1 = fmod(ph1_1 + w1 * rng[mi], 2. * np.pi)
    az1 = w0 * rng[mi] * r0 / np.tan(a) - \
          r1 * np.sin(a) * np.sin(w1 * rng[mi] + ph1_1)

    # keep going through 'a' positions on the second chain, looking for the 
    # smallest distance with the first a-position on chain 1, until the sign 
    # of the distance switches
    # start with the residue on the second chain that is "close-ish" to the 
    # first 'a' on the second chain
    if p_ap: # chains are parallel
        n = int(np.round((zoff - az1) * np.tan(a) / w0 / r0))
    else: # chains are antiparallel
        n = int(np.round((az1 - zoff) * np.tan(a) / w0 / r0))

    sgn = np.nan
    zaa = np.inf
    flag = 0
    # if the loop does not break within a couple of iterations, then we have 
    # a very bad fit and it does not matter anyway
    for count in range(100):
        # try up and down
        for ni in [n - count, n + count]:
            aph1_2 = ph1_2 + w1 * ni
            # though phase changes sign, to determine whether something is an 
            # 'a' or not, we still need the original phase
            if getHeptadPos(aph1_2, True) != 0:
                continue
            if p_ap: # chains are parallel
                az2 = zoff + w0 * ni * r0 / np.tan(a) - \
                      r1 * np.sin(a) * np.sin(w1 * ni + ph1_2)
            else: # chains are antiparallel
                az2 = zoff - w0 * ni * r0 / np.tan(a) + \
                      r1 * np.sin(a) * np.sin(w1 * ni + ph1_2)
            if np.abs(zaa) > np.abs(az2 - az1):
                zaa = az2 - az1
            if np.isnan(sgn) and np.sign(az2 - az1) != 0:
                sgn = np.sign(az2 - az1)
            # make sure both positive and negative Zaa' are tried, or if  
            # zero is found, that's obviously the lowest possible value
            if sgn * np.sign(az2 - az1) <= 0:
                flag = 1
                break
        if flag:
            break
    return zaa


def angleDiff(a, b):
    """Return wrapped differences between angles in two arrays.

    Parameters
    ----------
    a : float or np.array
        First angle or array of first angles (in radians).
    b : float or np.array
        Second angle or array of second angles (in radians), with which to 
        compute the wrapped difference from the corresponding first angle(s).

    Returns
    -------
    d : np.array
        Wrapped differences (in radians) between the angles in a and b.
    """
    d = fmod(fmod(a, 2. * np.pi) - fmod(b, 2. * np.pi), 2. * np.pi)
    if type(d) is np.float64 and d > np.pi:
        d -= 2. * np.pi
    elif type(d) is not np.float64:
        d[d > np.pi] = d[d > np.pi] - 2. * np.pi
    return d


def canonicalPhases(ind, angle_unit='degree'):
    """Return canonical phases of positions a-g indexed by an index array.

       The canonical phases are:
       [41.0, 95.0, 146.0, 197.0, 249.0, 300.0, 351.0]
       corresponding to {'c', 'g', 'd', 'a', 'e', 'b', 'f'}, respectively.

    Parameters
    ----------
    ind : np.array [7]
        Index array for the canonical phases.
    angle_unit : str
        The unit of angles to be returned by this function, either 
        'degree' or 'radian'.

    Returns
    -------
    ph : np.array [7]
        The canonical phases of positions a-g indexed by the array ind.
    """
    # in the order a-g
    median_phases = np.array([197.0, 300.0, 41.0, 146.0, 249.0, 351.0, 95.0])
    if angle_unit == 'radian':
        median_phases *= np.pi / 180.
    return median_phases[ind]


def crickEQ(r0, r1, w0, w1, a, ph0, ph1, t, bbtype):
    """Evaluate the Crick equations for x, y, and z given Crick parameters.

    Parameters
    ----------
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
    ph0 : float
        Helical phase angle (in radians) of the superhelix.
    ph1 : float
        Helical phase angle (in radians) of the minor helix.
    t : np.array [N]
        Values of the parameter t at which to evaluate the Crick equation.
    bbtype : str
        The type of backbone to generate. It should be one of the following 
        strings:
            'ca' -- a backbone containing only alpha carbon (Ca) atoms
            'gly' -- a backbone containing N, Ca, C, O, and H atoms
            'ala' -- a backbone containing N, Ca, C, O, Cb, and H atoms

    Returns
    -------
    xyz : np.array [N_atoms x 3]
        Cartesian coordinates (in Angstroms) of the coiled coil atoms 
        parameterized by the values of t.
    """
    if bbtype == 'ca':
        res_natoms = 1
        ca_idx = 0
    elif bbtype == 'gly':
        res_natoms = 5
        ca_idx = 1
    elif bbtype == 'ala':
        res_natoms = 6
        ca_idx = 1
    bb_coords = np.array([[[-0.883, -0.706, -0.908],   # N
                           [ 0.000,  0.000,  0.000],   # CA
                           [-0.721,  0.822,  1.056],   # C
                           [-0.312,  0.880,  2.215],   # O
                           [ 0.925,  0.914, -0.804],   # CB
                           [-0.901, -0.414, -1.843]]]) # H
    # since the x-axis of bb_coords lies along the shortest line segment 
    # between the superhelical axis and the alpha-carbon, the other two 
    # axes need to be inverted in the case of a negative w1 angle, since 
    # the direction from N-terminus to C-terminus is reversed
    if np.sign(w1) < 0:
        bb_coords[:, :, 1] *= -1.
        bb_coords[:, :, 2] *= -1.
    
    xyz = np.zeros((len(t), res_natoms, 3))
    # Crick exactly, as published in Acta Cryst. (1953). 6, 685
    xyz[:, ca_idx, 0] = \
        r0 * np.cos(w0 * t + ph0) + \
        r1 * np.cos(w0 * t + ph0) * np.cos(w1 * t + ph1) - \
        r1 * np.cos(a) * np.sin(w0 * t + ph0) * np.sin(w1 * t + ph1)
    xyz[:, ca_idx, 1] = \
        r0 * np.sin(w0 * t + ph0) + \
        r1 * np.sin(w0 * t + ph0) * np.cos(w1 * t + ph1) + \
        r1 * np.cos(a) * np.cos(w0 * t + ph0) * np.sin(w1 * t + ph1)
    xyz[:, ca_idx, 2] = \
        w0 * r0 * t / np.tan(a) - \
        r1 * np.sin(a) * np.sin(w1 * t + ph1)
    # impute other backbone coordinates if necessary
    if res_natoms > 1:
        c0, s0 = np.cos(w0 * t + ph0), np.sin(w0 * t + ph0)
        c1, s1 = np.cos(w1 * t + ph1), np.sin(w1 * t + ph1)
        R = np.array([[c0 * c1 - s0 * s1 * np.cos(a), 
                       s0 * c1 + c0 * s1 * np.cos(a), -s1 * np.sin(a)], 
                      [-c0 * s1 - s0 * c1 * np.cos(a), 
                       -s0 * s1 + c0 * c1 * np.cos(a), -c1 * np.sin(a)], 
                      [-s0 * np.sin(a), c0 * np.sin(a), 
                       np.ones_like(t) * np.cos(a)]])
        R = R.transpose((2, 0, 1))
        xyz = np.einsum('ijk,ikl->ijl', bb_coords, R) + \
              xyz[:, 1].reshape((-1, 1, 3))
    return xyz.reshape((-1, 3))


def getHeptadPos(ph1, return_int=False):
    """Return the heptad position corresponding to phase ph1.

    Parameters
    ----------
    ph1 : float
        Angle ph1 for which to return the heptad position.
    return_int : bool
        If True, return an integer index corresponding to the heptad position 
        instead of a character.
    
    Returns
    -------
    hp : str or int
        The heptad position corresponding to phase ph1, either as a character 
        from a-g or an integer from 0-6.
    """
    meds = canonicalPhases(np.arange(7), 'radian')
    hps = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g'])

    ph1 = fmod(ph1, 2. * np.pi)
    if np.any(ph1 == meds):
        if return_int:
            return np.argwhere(ph1 == meds)
        else:
            return hps[ph1 == meds]

    # sort phases in the order they appear on the helical wheel
    si = np.argsort(meds)
    meds = meds[si]
    hps = hps[si]

    for i in range(len(meds)):
        pin = i - 1
        nin = i + 1
        if i == 0:
            pin = len(meds) - 1
        if i == len(meds) - 1:
            nin = 0
        lb = fmod(angleDiff(meds[pin], meds[i]) / 2. + meds[i], 2. * np.pi) 
        ub = fmod(angleDiff(meds[nin], meds[i]) / 2. + meds[i], 2. * np.pi)
        if angleDiff(ph1, lb) > 0 and angleDiff(ub, ph1) > 0:
            if return_int:
                return 'abcdefg'.index(hps[i])
            else:
                return hps[i]
    raise ValueError('{} {}'.format(ph1, meds))


def retrieve_name(var):
    """Retrieve name of variable.

    Parameters
    ----------
    var : ANY
        Variable for which to retrieve the name.

    Returns
    -------
    varname : str
        Name of variable.
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if 
            var_val is var][0]


def fmod(x, m):
    """Returns x - floor(x / m) * y if y != 0.

    Parameters
    ----------
    x : float
        Number to compute modulo m.
    m : float
        Modulus of the computation.

    Returns
    -------
    y : float
        x - floor(x / m) * m
    """
    return x - np.floor(x / m) * m
