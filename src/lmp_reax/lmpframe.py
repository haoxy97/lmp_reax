#!/usr/bin/env python3

from pathlib import Path
import re
from collections import Counter
from typing import Tuple, List
from itertools import repeat
from openbabel import openbabel as ob
from lmp_env import type2element_str_dict, type2element_id_dict, MAX_SMILE_MOLE_NUM, METAL, SKIP_FRAME
import array
import gzip


convector = ob.OBConversion()
convector.SetOutFormat('can')
OB_atom_tuple = []
for atom_id in None, 1, 2, 3, 4:
    if atom_id is None:
        atom_temp = None
    else:
        atom_temp = ob.OBAtom()
        atom_temp.SetAtomicNum(type2element_id_dict[atom_id])
    OB_atom_tuple.append(atom_temp)
OB_atom_tuple = tuple(OB_atom_tuple)


def order_tuple(a: int, b: int) -> tuple:
    """
    sort the tuple by values
    :param a: val1
    :param b: val2
    :return: order new tuple
    """
    if a > b:
        return b, a
    else:
        return a, b


def init_BondFrame(lmp_data_path: str):
    """init BondFrame.type and BondFrame.element,
    tuples which map from an id to the type of element"""
    context = Path(lmp_data_path).read_text()
    context = re.sub(r"#.+", "", context)  # remove comment in data file
    n_atoms = re.search(r'(\s*(\d+)\s*atoms\n)', context).groups()[1]
    n_atoms = int(n_atoms)
    atom_context = re.split(r'Atoms.*\n\n', context)[-1]

    type_list = [int(item.split()[1])
                 for item in atom_context.split('\n')
                 if len(item) > 0]
    if len(type_list) != n_atoms:
        print('%s: atom number not match' % lmp_data_path)
        raise FileNotFoundError

    type_list = [0] + type_list
    element_list = [type2element_str_dict[x] for x in type_list]
    BondFrame.type_tuple = tuple(type_list)
    BondFrame.element_tuple = tuple(element_list)
    BondFrame.n_atoms = len(type_list)
    if METAL[0]:
        BondFrame.metal_index_tuple = tuple([i for i, x in enumerate(BondFrame.type_tuple) if x == METAL_ID[0]])

    # init open babel atom tuple


class BondFrame:
    """object stores status of MD trace at a certain moment"""
    type_tuple = tuple()
    element_tuple = tuple()
    metal_index_tuple = tuple()

    n_atoms = 0
    # C:1, H:2, O:3
    # bond value formula: C-H,C-O,O-H
    bond_value_dict = {'11': 0, '12': 100, '21': 100, '13': 10, '31': 10,
                       '22': 0, '23': 1, '32': 1, '33': 0, '00': 0}

    def __init__(self, context: bytes):
        """
        context: bytes of bond frame, from splitting lz4 file
        contain Gst(ghost) 0 atom to solve index [0 - 1] enharmonic of python
        """
        if self.element_tuple is None:
            print('Bond Frame must init from data first')
            raise TypeError
        self.n_atoms = len(self.type_tuple)
        context = re.split(b'\n', context)

        self.step = int(context[0].split()[-1])
        
        raw_bond_list=[(0,0)]
        for val_line in context[1:-1]:
            val_line=val_line.split()
            raw_bond_list.append([int(x) for x in (val_line[0],*val_line[3:3+int(val_line[2])])])
        # simple bond version
        # raw_bond_list=[]
        # context[-1] = b"0 0"
        # raw_bond_list = list(map(lambda val: [int(val_item) for val_item in val.split()], context[1:]))
        raw_bond_list.sort(key=lambda val_x: val_x[0])

        self.atom_neighbor_dict = dict(zip(range(self.n_atoms), repeat(None)))
        connect_list = []
        for item in raw_bond_list:
            if len(item) < 2:
                continue
            a = item[0]
            bond_list = [order_tuple(a, x) for x in item[1:]]
            # self.neighbor_dict[a] = bond_list
            self.atom_neighbor_dict[a] = tuple(item[1:])
            connect_list += bond_list
        # connect set is used for detect rxn occurs
        self.connect_set = set(connect_list)  # use set to remove repeat item
        self.bond_counter = self.build_bond_counter()
        self.specie_id_list, self.specie_counter = self.build_atom2specie_list()
        del connect_list[:]
        del self.atom_neighbor_dict

    def parse_specie_name(self, specie_ids: set, formula_only=False) -> str:
        """use openBabel to parse atom connection table to Canonical SMILES format"""
        specie_ids = frozenset(specie_ids)
        # ignore the reaction: H2O__2O-H,H-H -> H2O__2O-H
        specie_counter = Counter([self.element_tuple[item]
                                  for item in specie_ids])
        specie_name = []
        for element in type2element_str_dict[1], type2element_str_dict[2], \
                       type2element_str_dict[3], type2element_str_dict[4]:
            num = specie_counter[element]
            if num > 0:
                specie_name.append(element)
            if num > 1:
                specie_name.append("%d" % num)
        formula = "%s" % ("".join(specie_name))
        # small or too large atom numbers , only use chemical formula to boost calculate.
        if len(specie_ids) < 4 or formula_only:
            return formula
        bond_val = 0
        # neighbor_list = set(sum([self.neighbor_dict[x] for x in specie_ids], []))
        neighbor_list = set(sum([list(map(lambda y: order_tuple(x, y), self.atom_neighbor_dict[x]))
                                 for x in specie_ids], []))
        for bond in neighbor_list:
            bond0, bond1 = bond
            bond_val += self.bond_value_dict.get('%d%d' % (self.type_tuple[bond0], self.type_tuple[bond1]), 0)

        if len(specie_ids) > MAX_SMILE_MOLE_NUM:
            return "%s__%04d" % (formula, bond_val)
            # return "%d" % len(specie_ids)
        mol = ob.OBMol()
        # local_id_dict[key: atom global id]:[val: local id(1,2,3...)]
        local_id_dict = dict([(x, i + 1) for i, x in enumerate(specie_ids)])
        # use OpenBabel to get smiles string
        for atom in specie_ids:
            mol.AddAtom(OB_atom_tuple[self.type_tuple[atom]])

        for bond in neighbor_list:
            bond0, bond1 = bond
            start = local_id_dict[bond0]
            end = local_id_dict[bond1]
            mol.AddBond(start, end, 1)
        # use OpenBabel.SSSR get ring info
        ring_Counter = Counter([x.Size() for x in mol.GetSSSR()])
        ring_str = "".join(["r%s-%d" % (x, y) for x, y in ring_Counter.items()])
        specie_str = "%s__%04d%s#%s" % (
            formula, bond_val, ring_str, convector.WriteString(mol).rstrip('\t\n').replace(r'@', ''))
        return specie_str


    def build_atom2specie_list(self) -> Tuple[List[str], Counter]:
        "use breath first search"
        atom2specie_list = [""] * self.n_atoms
        visited_list = [False] * self.n_atoms
        frame_specie_list = []
        atom_stack = array.array('i', [-1])
        for index in range(1, self.n_atoms):
            if not visited_list[index]:
                specie_ids = []
                atom_stack.append(index)
                while True:
                    atom = atom_stack.pop()
                    if atom < 0:
                        atom_stack.append(-1)
                        break
                    elif visited_list[atom]:
                        continue
                    specie_ids.append(atom)
                    if self.atom_neighbor_dict[atom] is not None:
                        for neighbor in self.atom_neighbor_dict[atom]:
                            if not visited_list[neighbor]:
                                atom_stack.append(neighbor)
                    visited_list[atom] = True
                specie_ids = set(specie_ids)
                specie_name = self.parse_specie_name(specie_ids)
                frame_specie_list.append(specie_name)  # counter species
                for i in specie_ids:  # update atom 2 specie_list
                    atom2specie_list[i] = specie_name

        return atom2specie_list, Counter(frame_specie_list)

    def build_bond_counter(self):

        return Counter(['%d%d' % order_tuple(self.type_tuple[item[0]],
                                             self.type_tuple[item[1]])
                        for item in self.connect_set])

    def __getitem__(self, index):
        """
        index by id
        return element
        """
        return self.element_tuple[index]

    def __len__(self):
        return self.n_atoms

    def __repr__(self):
        return "Bond frame at step %d" % self.step


def gz_2_block(zip_file: str,block_size=1000):
    with gzip.open(zip_file, 'rb') as f:
        content = f.read()
    context=re.split(rb'# \n',re.sub(rb'#\n#\sNumber.*\n'+4*b'#.*\n',b'', content))[1:-1]
    frame_block_list = []
    for i in range(0, len(context), block_size):
        frame_block_list.append(context[i:i + block_size])
    return frame_block_list

