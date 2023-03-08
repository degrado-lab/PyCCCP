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

import numpy as np

from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

def coordsToPDB(xyz, chains, pdb_path, bbtype='ca'):
    """Output coordinates to PDB file.

    Parameters
    ----------
    XYZ : np.array [N_atoms x 3]
        Coordinates (in Angstroms) of backbone atoms.
    chains : np.array [N_atoms]
        Single-letter chain IDs of backbone atoms.
    pdb_path : str
        Path at which to output a PDB file.
    bbtype : str
        The type of backbone to generate. It should be one of the following 
        strings:
            'ca' -- a backbone containing only alpha carbon (Ca) atoms
            'gly' -- a backbone containing N, Ca, C, O, and H atoms
            'ala' -- a backbone containing N, Ca, C, O, Cb, and H atoms     
    """
    if bbtype == 'ca':
        res_natoms = 1
        resname = 'GLY'
        names = ['CA']
        fullnames = [' CA ']
        elements = ['C']
    elif bbtype == 'gly':
        res_natoms = 5
        resname = 'GLY'
        names = ['N', 'CA', 'C', 'O', 'H']
        fullnames = ['  N ', ' CA ', '  C ', '  O ', '  H ']
        elements = ['N', 'C', 'C', 'O', 'H']
    elif bbtype == 'ala':
        res_natoms = 6
        resname = 'ALA'
        names = ['N', 'CA', 'C', 'O', 'CB', 'H']
        fullnames = ['  N ', ' CA ', '  C ', '  O ', ' CB ', '  H ']
        elements = ['N', 'C', 'C', 'O', 'C', 'H']
    
    struct = Structure('crickbb')
    model = Model(0)
    atom_num = 0
    resnum = 1
    for chid in np.unique(chains):
        chain = Chain(chid)
        for i in range((chains == chid).sum() // res_natoms):
            residue = Residue((' ', resnum, ' '), resname, 'A')
            for j, name in enumerate(names):
                atom = Atom(name, xyz[atom_num], 0.0, 1.0, ' ', 
                            fullnames[j], atom_num + 1, elements[j])
                atom_num += 1
                residue.add(atom)
            resnum += 1
            chain.add(residue)
        model.add(chain)
    struct.add(model)
    io = PDBIO()
    io.set_structure(struct)
    io.save(pdb_path)
