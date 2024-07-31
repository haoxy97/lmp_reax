#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import pickle
from itertools import chain
import numpy as np
from functools import lru_cache
import bisect
from dataclasses import dataclass
from math import ceil
from typing import List
from .lmpframe import init_BondFrame, BondFrame
from .lmp_env import METAL


@dataclass
class FrameTrace:
    """
    object stores list of block status info, which contains summed or average info of frame batch
    """
    avg: int
    step: List[int]
    bond: List[Counter]
    specie: List[Counter]
    rxn: List[Counter]

    def __repr__(self):
        return "FrameTrace %d to %d" % (self.step[0], self.step[-1])

    def __len__(self):
        return len(self.step)

    def __add__(self, other):
        return FrameTrace(self.avg, self.step + other.step, self.bond + other.bond, self.specie + other.specie,
                          self.rxn + other.rxn)

    def to_pickle(self, save_name=None):
        if save_name is None:
            save_name = 'trace/%s.pickle' % self.step[0]
        pickle.dump(self, Path(save_name).open('wb'))


class FrameTraceInitializer:
    """
    init frame trace from bondFrame list
    """

    def __init__(self, batch: List[BondFrame], sample_avg=100):
        self.batch = batch
        self.sample_avg = sample_avg
        self.n_frame = len(batch)

    def to_FrameTrace(self):
        return FrameTrace(self.sample_avg, self.step_trace.tolist(),
                          self.bond_trace, self.specie_trace, self.rxn_trace)

    @property
    @lru_cache(maxsize=None)
    def rxn_trace(self) -> List[Counter]:
        """split rxn list info constant width time section list"""

        output = []
        index_b = np.array(range(1, self.n_frame, 1), dtype=int)
        index_a = index_b - 1
        raw_output = list(map(self.compare_frame, index_a, index_b))
        raw_output = [item for item in raw_output if item is not None]
        if len(raw_output) < 1 or len(self.step_trace)<2:
            return [Counter()] * len(self.step_trace)
        step_section_list = self.step_trace + (self.step_trace[1] - self.step_trace[0]) // 2

        rxn_step_list, rxn_list = list(zip(*raw_output))
        subset_index_a, subset_index_b = 0, 0

        # to count the times of rxn from t_a to t_b,
        # use bisect to find the index of t_a and t_b in rxn_list
        for step_end in step_section_list:
            subset_index_b = bisect.bisect_left(rxn_step_list, step_end)
            if subset_index_b > subset_index_a:
                output.append(Counter(sum(rxn_list[subset_index_a:subset_index_b], [])))
            else:
                output.append(Counter())
            subset_index_a = subset_index_b
        return output

    @property
    @lru_cache(maxsize=None)
    def step_trace(self):
        output = np.array(list(map(lambda x: self.batch[x].step, range(0, self.n_frame, self.sample_avg)))
                          , dtype=int)
        if len(output)<2:
            return output
        output += (output[1] - output[0]) // 2  # correlate: old frame step to the middle step of old and new
        return output

    @property
    @lru_cache(maxsize=None)
    def specie_trace(self):
        raw_output = list(map(lambda x: self.batch[x].specie_counter, range(self.n_frame)))
        output = []
        for index in range(0, len(raw_output), self.sample_avg):
            sub_output = raw_output[index:index + self.sample_avg]
            avg = len(sub_output)
            if avg < 2:
                output.append(Counter())
            else:
                sub_output = sum(sub_output, Counter())
                sub_counter = dict((key, ceil(val / avg)) for key, val in sub_output.items())
                sub_counter = Counter(sub_counter)
                output.append(sub_counter)
        return output

    @property
    @lru_cache(maxsize=None)
    def bond_trace(self):
        raw_output = list(map(lambda x: self.batch[x].bond_counter, range(self.n_frame)))
        output = []
        for index in range(0, len(raw_output), self.sample_avg):
            sub_output = raw_output[index:index + self.sample_avg]
            avg = len(sub_output)
            if avg < 2:
                output.append(Counter())
            else:
                sub_output = sum(sub_output, Counter())
                sub_counter = dict((key, ceil(val / avg)) for key, val in sub_output.items())
                sub_counter = Counter(sub_counter)
                output.append(sub_counter)
        return output

    def compare_frame(self, index_a: int, index_b: int):
        """
        find the bond change moment between two frame
        inner bond is marked as "A--B", and in left is break, in right is form
        :return: list of reaction
        """
        old_frame = self.batch[index_a]
        new_frame = self.batch[index_b]

        # use bond and specie counter check, speed up compare
        if old_frame.bond_counter == new_frame.bond_counter:
            if old_frame.specie_counter == new_frame.specie_counter:
                return None

        reaction_info = []

        if METAL[0]:
            for index in old_frame.metal_index_tuple:
                new = new_frame.specie_id_list[index]
                old = old_frame.specie_id_list[index]
                if new == old:
                    continue
                else:
                    reaction_info.append("%s->%s" % (old, new))
            return old_frame.step, reaction_info

        break_bond_set = old_frame.connect_set.difference(new_frame.connect_set)
        forms_bond_set = new_frame.connect_set.difference(old_frame.connect_set)

        # case (num of bond broken, num of bond formation)
        if len(forms_bond_set) == 0:
            if len(break_bond_set) > 0:  # case (n,0)
                for a, b in break_bond_set:
                    rxn_str = parse_bond_break(old_frame, new_frame, a, b)
                    reaction_info.append(rxn_str)
            else:
                return None
        else:
            if len(break_bond_set) == 0:  # case (0,n)
                for a, b in forms_bond_set:
                    rxn_str = parse_bond_forms(old_frame, new_frame, a, b)
                    reaction_info.append(rxn_str)
            else:
                # case (1,1)
                if len(break_bond_set) == 1 and len(forms_bond_set) == 1:

                    a0, b0 = next(iter(break_bond_set))
                    a1, b1 = next(iter(forms_bond_set))

                    bond_atom_set = {a0, b0, a1, b1}

                    if len(bond_atom_set) == 3:
                        rxn_str = parse_bond_break_forms(old_frame, new_frame, a0, b0, a1, b1)
                        reaction_info.append(rxn_str)
                    if len(bond_atom_set) == 4:
                        rxn_str = parse_bond_break(old_frame, new_frame, a0, b0)
                        reaction_info.append(rxn_str)
                        rxn_str = parse_bond_forms(old_frame, new_frame, a1, b1)
                        reaction_info.append(rxn_str)
                # case (n,n)
                else:
                    common_atom_set = set(chain(*break_bond_set)).intersection(set(chain(*forms_bond_set)))
                    if len(common_atom_set) > 0:
                        for atom in common_atom_set:
                            break_bond1, forms_bond1 = None, None
                            for break_bond in break_bond_set:
                                if atom in break_bond:
                                    break_bond1 = break_bond
                                    break
                            for forms_bond in forms_bond_set:
                                if atom in forms_bond:
                                    forms_bond1 = forms_bond
                                    break
                            if break_bond1 is None or forms_bond1 is None:
                                continue
                            else:

                                a0, b0 = break_bond1
                                a1, b1 = forms_bond1
                                bond_atom_set = {a0, b0, a1, b1}

                                if len(bond_atom_set) < 4:
                                    break_bond_set.remove(break_bond1)
                                    forms_bond_set.remove(forms_bond1)

                                if len(bond_atom_set) == 3:
                                    rxn_str = parse_bond_break_forms(old_frame, new_frame, a0, b0, a1, b1)
                                    reaction_info.append(rxn_str)

                    for a, b in break_bond_set:
                        rxn_str = parse_bond_break(old_frame, new_frame, a, b)
                        reaction_info.append(rxn_str)
                    for a, b in forms_bond_set:
                        rxn_str = parse_bond_forms(old_frame, new_frame, a, b)
                        reaction_info.append(rxn_str)

        if len(reaction_info) < 1:
            return None
        # remove connect set after compare for less mem usage
        old_frame.connect_set = None
        return old_frame.step, reaction_info


def parse_bond_break(old_frame: BondFrame, new_frame: BondFrame, a: int, b: int):
    reactant = old_frame.specie_id_list[a]
    specie_a = new_frame.specie_id_list[a]
    specie_b = new_frame.specie_id_list[b]

    products = "%s+%s" % (specie_a, specie_b)

    if specie_a == specie_b:
        # inner case
        if specie_a.split("_")[0] == reactant.split("_")[0]:
            reactant = "%s+%s--%s" % (reactant, old_frame[a], old_frame[b])
            products = specie_a
    return "%s->%s" % (reactant, products)


def parse_bond_forms(old_frame: BondFrame, new_frame: BondFrame, a: int, b: int):
    products = new_frame.specie_id_list[a]
    specie_a = old_frame.specie_id_list[a]
    specie_b = old_frame.specie_id_list[b]

    reactant = "%s+%s" % (specie_a, specie_b)

    if specie_a == specie_b:
        # inner case
        if specie_a.split("_")[0] == products.split("_")[0]:
            reactant = specie_a
            products = '%s+%s--%s' % (products, new_frame[a], new_frame[b])

    return "%s->%s" % (reactant, products)


def parse_bond_break_forms(old_frame: BondFrame, new_frame: BondFrame, a0: int, b0: int, a1: int, b1: int):
    if a0 in (a1, b1):
        pass
    if b0 in (a1, b1):
        a0, b0 = b0, a0

    if a0 == b1:
        a1, b1 = b1, a1

    reactant = "%s+%s" % (old_frame.specie_id_list[a0], old_frame.specie_id_list[b1])
    products = "%s+%s" % (new_frame.specie_id_list[a0], new_frame.specie_id_list[b0])
    return "%s->%s" % (reactant, products)

