#!/usr/bin/env python3

from pathlib import Path
import re
from collections import Counter
from typing import Tuple, List
from itertools import repeat
from openbabel import openbabel as ob
from lmp_env import type2element_str_dict, type2element_id_dict, MAX_SMILE_MOLE_NUM, METAL, METAL_ID, SKIP_FRAME
import lz4.frame
import array

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
        context[-1] = b"0 0"

        raw_bond_list = list(map(lambda val: [int(val_item) for val_item in val.split()], context[1:]))
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

        if METAL[0]:
            self.specie_id_list, self.specie_counter = self.build_atom2specie_list_metal()
        else:
            self.specie_id_list, self.specie_counter = self.build_atom2specie_list()

        if METAL[0]:
            del self.connect_set
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

    def build_atom2specie_list_metal(self) -> Tuple[List[str], Counter]:
        metal_id = METAL_ID[0]
        "use breath first search, metal must be No.4 type atom"
        atom2specie_list = [""] * self.n_atoms
        visited_list = [False] * self.n_atoms
        frame_specie_list = []

        # 1. parse metal atom neighbor, parse_particle component
        for index in range(1, self.n_atoms):
            if self.type_tuple[index] != metal_id:
                continue
            visited_list[index] = True
            if self.atom_neighbor_dict[index] is not None:
                metal_neighbor = [x for x in self.atom_neighbor_dict[index] if self.type_tuple[x] != metal_id]
                metal_stack = array.array('i', [-1])
                particle_stack = array.array('i', [-1])
                metal_neighbor_name_list = []
                metal_local_visited_list = []
                for non_metal in metal_neighbor:
                    metal_ids = []
                    metal_stack.append(non_metal)

                    prticle_ids = []
                    particle_stack.append(non_metal)

                    # parse species in metals
                    while True:
                        atom = particle_stack.pop()
                        if atom < 0:
                            particle_stack.append(-1)
                            break
                        elif visited_list[atom]:
                            continue
                        prticle_ids.append(atom)
                        visited_list[atom] = True

                        if self.atom_neighbor_dict[atom] is not None:
                            for neighbor in self.atom_neighbor_dict[atom]:
                                if neighbor not in metal_local_visited_list:
                                    if self.type_tuple[neighbor] != metal_id:
                                        if self.type_tuple[neighbor] != 1: # for 'C' id 1
                                            particle_stack.append(neighbor)

                    prticle_ids = set(prticle_ids)
                    if len(prticle_ids) > 0:
                        frame_specie_list.append('#metal#(%s)' % self.parse_specie_name(prticle_ids, formula_only=True))

                    # parse metal neighbors
                    while True:
                        atom = metal_stack.pop()
                        if atom < 0:
                            metal_stack.append(-1)
                            break
                        elif atom in metal_local_visited_list:
                            continue
                        metal_ids.append(atom)
                        metal_local_visited_list.append(atom)

                        if self.atom_neighbor_dict[atom] is not None:
                            for neighbor in self.atom_neighbor_dict[atom]:
                                if neighbor not in metal_local_visited_list:
                                    if self.type_tuple[neighbor] != metal_id:
                                        if self.type_tuple[neighbor] != 1: # for 'C' id 1
                                            metal_stack.append(neighbor)

                    metal_ids = set(metal_ids)
                    if len(metal_ids) > 0:
                        if len(metal_ids) > 1:
                            metal_neighbor_name_list.append(
                                '(%s)' % self.parse_specie_name(metal_ids, formula_only=True))
                        else:
                            metal_neighbor_name_list.append(self.element_tuple[metal_ids.__iter__().__next__()])

                specie_name = "%s%s" % (self.element_tuple[index],
                                        "".join(["%s%d" % (x[0], x[1])
                                                 for x in sorted(Counter(metal_neighbor_name_list).items())]))
            else:
                specie_name = self.element_tuple[index]
            atom2specie_list[index] = specie_name
            frame_specie_list.append(specie_name)

        # 3. parse env molecule
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
                if len(specie_ids) > 0:
                    specie_name = self.parse_specie_name(specie_ids, formula_only=True)
                    if len(specie_name) > 0:
                        frame_specie_list.append(specie_name)  # counter species
                    for i in specie_ids:  # update atom 2 specie_list
                        atom2specie_list[i] = specie_name

        return atom2specie_list, Counter(frame_specie_list)

    def build_atom2specie_list(self) -> Tuple[List[str], Counter]:
            atom2specie_list = [None] * self.n_atoms
            frame_specie_list = []
            atom_stack=[] # use common list to avoid memory leak
            for index in range(1, self.n_atoms):
                if atom2specie_list[index] is None:
                    specie_ids = []
                    atom_stack.append(index)
                    while atom_stack:
                        atom = atom_stack.pop()
                        if atom2specie_list[atom] is not None:
                            continue
                        specie_ids.append(atom)
                        if self.atom_neighbor_dict[atom] is not None:
                            for neighbor in self.atom_neighbor_dict[atom]:
                                if atom2specie_list[neighbor] is None:
                                    atom_stack.append(neighbor)
                        atom2specie_list[atom] = "" # mark as explored
                    specie_ids = set(specie_ids) # remove duplicate ids
                    specie_name = self.parse_specie_name(specie_ids)
                    frame_specie_list.append(specie_name)  # counter species
                    for i in specie_ids:  # update atom specie_list to species name
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


def lz4_2_block(lz4_file: str, block_size=1000) -> List:
    """split raw lz4 data into small batches(which calls blocks),
    lz4 data contains bond info in the order of time series"""
    # use block to split lz4 batch to reduce mem usage
    with lz4.frame.open(lz4_file, mode='r') as f:
        lz4_context = re.split(rb'# \n', f.read())

    # -2 for drop last[-1] and second last[-2],
    # [-1] for empty line and [-2] the for duplicate of next batch[0]
    lz4_context = [item for item in lz4_context if len(item) > 0][:-2][::SKIP_FRAME[0]]
    frame_block_list = []
    if block_size < 0:
        return [lz4_context]
    for i in range(0, len(lz4_context), block_size):
        frame_block_list.append(lz4_context[i:i + block_size])
    return frame_block_list


if __name__ == '__main__':
    from pprint import pprint
    import time
    from lmpdatatrace import FrameTraceInitializer

    # metal system
    import lmp_env

    lmp_env.SKIP_FRAME[0] = 5

    t0 = time.time()
    # init_BondFrame('example/aluminum/Al_30x30x40_ox_10_water_rho50_depth_50.data')
    # frames_blocks = lz4_2_block("example/aluminum/part_bond/002_part.bond.lz4", block_size=100)

    print(time.time() - t0)
    t0 = time.time()
    # lmp_env.METAL_ID[0] = 4
    # lmp_env.METAL_NAME[0] = 'Fe'
    # lmp_env.type2element_str_dict[lmp_env.METAL_ID[0]] = lmp_env.METAL_NAME[0]

    # init_BondFrame('example/OH/Al_20x20x0_ox_0_water_rho50_depth_40.data')
    # frames_blocks = lz4_2_block("example/OH/bond/001.bond.lz4", block_size=100)
    # for index1, block in enumerate(frames_blocks):
    #     metal = BondFrame(block[-1])
    #     print(metal.specie_counter)
    #     break
    #     frame_block = list(map(BondFrame, block[-40:]))
    #     spam = FrameTraceInitializer(batch=frame_block, sample_avg=20).to_FrameTrace()
    #     pprint(spam.specie)
    #     break
    # print(time.time() - t0)

    # CHO system
    if 1:
        # init_BondFrame('example/5A2/box_46_rho_51.data'/)
        # frames_blocks = lz4_2_block("example/5A2/bond/0002.bond.lz4", block_size=100)
        init_BondFrame('example/methane/inp.data')
        frames_blocks = lz4_2_block("example/methane/bond/0500.bond.lz4", block_size=100)
        t0 = time.time()
        for index1, block in enumerate(frames_blocks):
            a=BondFrame(block[-1])
            print(a.specie_counter)
            break
            frame_block = list(map(BondFrame, block[:40]))
            spam = FrameTraceInitializer(batch=frame_block, sample_avg=20).to_FrameTrace()
            pprint(spam.specie)
            break
        print(time.time() - t0)
