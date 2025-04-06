#!/usr/bin/env python3
from dataclasses import dataclass
import numpy as np
from typing import Tuple, List, Type, Union
from scipy.spatial.distance import pdist, squareform
from itertools import repeat
from .env import EnvFrame
from collections import Counter
from utility import timing_decorator
from collections.abc import Iterable
# here, we re impelement the Frame, not use the init frame method,
# add the code is for metal system only, suppose id 1 is metal
# use numpy pickle to save the data
# improve cache type, try use numpy array to store the data


@dataclass
class FrameEnv:
    n_atoms: int
    type_array: np.ndarray
    element_array: np.ndarray
    dis_sq_ref: np.ndarray
    METAL_ID = 1


class Frame:
    step: int
    specie_id_list: List[str]
    env: FrameEnv
    bond_value_dict = {'11': 0, '12': 100, '21': 100, '13': 10, '31': 10,
                       '22': 0, '23': 1, '32': 1, '33': 0, '00': 0}

    def __init__(self, step, cord, box, env):
        self.step = step
        self.box = box
        self.env = env
        # build neibor dict
        atom_neighbor_list = [list() for _ in range(self.env.n_atoms)]
        # connect_list = []
        id_array = np.arange(self.env.n_atoms)
        for i, item in enumerate(squareform(distance_sq_mat(cord, box) < self.env.dis_sq_ref)):
            item = id_array[item]
            if len(item) > 0:
                atom_neighbor_list[i] = item
                # connect_list += [(i, x) for x in item if i < x]
        self.specie_id_list = self.build_specie_list(atom_neighbor_list)

    def parse_specie(self, idxs):
        if not isinstance(idxs, Iterable):
            element_counter = Counter([self.env.type_array[idxs]])
        else:
            element_counter = Counter(self.env.type_array[idxs])
        return "".join(["%s%d" % (EnvFrame.type2element_str_dict[k],
                                  element_counter[k]) for k in sorted(element_counter.keys())])

    def build_specie_list(self, neighbor_list, metal_mark='#METAL#'):
        metal_id = self.env.METAL_ID
        metal_name = EnvFrame.METAL_NAME
        n_atoms = self.env.n_atoms
        type_array = self.env.type_array
        element_array = self.env.element_array
        specie_id_list = [None] * n_atoms
        marked_specie_id_list = []
        # we first mark the #metal#H2O1 with its ids(start with 2nd, 1st for record) to avoid repeat search, then we remove its ids in name
        # use marked_specie_id_list to record the marked ids.
        # for metal, not search its metal neighbors, record as metal(metal)n(X)m(Y)...
        for center_id, neighbor_idx in enumerate(neighbor_list):
            if type_array[center_id] == metal_id:
                name = [metal_name]
                metal_neighbor_count = 0
                non_metal_neighbor = []
                non_metal_name = []
                for idx in neighbor_idx:
                    if type_array[idx] == metal_id:
                        pass
                        # ignore the metal neighbor
                        # metal_neighbor_count += 1
                    else:
                        non_metal_neighbor.append(idx)

                if metal_neighbor_count > 0:
                    name.append("(%s)%d" % (metal_name, metal_neighbor_count))

                # update from deep search to limited 3 layer search
                # also check the inter neighbor connectivity
                for non_metal_id in non_metal_neighbor:
                    if specie_id_list[non_metal_id] is None:
                        # to be parsed

                        non_metal_id_neighbor = neighbor_list[non_metal_id]
                        non_metal_id_neighbor_types = type_array[non_metal_id_neighbor]
                        sub_neighbor = non_metal_id_neighbor[non_metal_id_neighbor_types != 1]
                        if len(sub_neighbor) == 0:
                            specie_id_list[non_metal_id] = '%s%s' % (metal_mark, element_array[non_metal_id])
                            non_metal_name.append(element_array[non_metal_id])
                        else:
                            if len(sub_neighbor) == 1:
                                sub_sub_neighbor = neighbor_list[sub_neighbor[0]].tolist()
                            else:
                                sub_sub_neighbor = list(set(np.concatenate([neighbor_list[x] for x in sub_neighbor])))
                            sub_sub_neighbor.remove(non_metal_id)
                            sub_sub_type = type_array[sub_sub_neighbor]
                            if np.sum(sub_sub_type) == metal_id * len(sub_sub_neighbor):
                                sub_stack = np.concatenate([sub_neighbor, [non_metal_id]])
                            else:
                                # append the sub_sub_neighbor
                                non_metal_sub = np.array(sub_sub_neighbor)[sub_sub_type != 1]
                                sub_stack = np.concatenate([sub_neighbor, non_metal_sub, [non_metal_id]])

                                # check the 3 layer is all metal
                                if len(non_metal_sub) == 1:
                                    sub_sub_sub_neighbor = neighbor_list[non_metal_sub[0]].tolist()
                                else:
                                    sub_sub_sub_neighbor = list(
                                        set(np.concatenate([neighbor_list[x] for x in non_metal_sub])))
                                # remove for the substituion of sub_sub_neighbor
                                sub_sub_sub_neighbor = [x for x in sub_sub_sub_neighbor if x not in sub_stack]
                                sub_sub_sub_type = type_array[sub_sub_sub_neighbor]
                                if np.sum(sub_sub_sub_type) != metal_id * len(sub_sub_sub_neighbor):
                                    print('warning: 3 layer not all metal')
                            sub_name = self.parse_specie(sub_stack)
                            non_metal_name.append("%s" % sub_name)
                            # -1 for ignore the non_metal_id
                            sub_name_with_mark = sub_name+'-%s' % metal_mark+'+'.join(str(x) for x in sub_stack[:-1])
                            marked_specie_id_list.append(sub_stack)
                            for atom in sub_stack:
                                specie_id_list[atom] = sub_name_with_mark
                    else:
                        specie = specie_id_list[non_metal_id]
                        if len(specie) == 1+len(metal_mark):
                            non_metal_name.append(element_array[non_metal_id])
                        else:
                            if str(non_metal_id) not in specie:
                                non_metal_name.append("%s" % specie.split('-')[0])

                if len(non_metal_name) > 0:
                    name_counter = Counter(non_metal_name)
                    sub_name = ''.join(['(%s)%d' % (k, name_counter[k])
                                        for k in sorted(name_counter.keys())])
                    name.append(sub_name)
                specie_id_list[center_id] = ''.join(name)
            else:
                if specie_id_list[center_id] is not None:
                    continue
                non_metal_neighbor = [None, center_id]
                sub_neighbor = []
                for _ in range(101):
                    atom = non_metal_neighbor.pop()
                    if atom is None:
                        break
                    if atom in sub_neighbor:
                        continue
                    sub_neighbor.append(atom)
                    atom_neighbor = neighbor_list[atom]
                    for atom_idx in atom_neighbor:
                        if atom_idx not in sub_neighbor and type_array[atom_idx] != metal_id:
                            non_metal_neighbor.append(atom_idx)
                if _ == 100:
                    print('max iter reached for parsing non-metal neighbors')
                name_counter = Counter(type_array[sub_neighbor])
                name = ''.join(['%s%d' % (EnvFrame.type2element_str_dict[k], name_counter[k])
                                for k in sorted(name_counter.keys())])
                for atom in sub_neighbor:
                    specie_id_list[atom] = name
        if len(marked_specie_id_list):
            for atom in set(np.concatenate(marked_specie_id_list)):
                specie_id_list[atom] = metal_mark+specie_id_list[atom].split('-')[0]
        return specie_id_list


def distance_sq_mat(cords: np.ndarray, box_length: np.ndarray) -> np.ndarray:
    # with box, support PBC
    # read from bond data, 3.6s / 1000 frames; calculate directly 19s / 1000 frames; with 8 cores
    dist_mat = None
    for dim in range(len(box_length)):
        dis_x = np.float32(pdist(cords[:, dim][:, np.newaxis]))
        dis_x_pbc = np.minimum(box_length[dim] - dis_x, dis_x)
        if dist_mat is None:
            dist_mat = np.square(dis_x_pbc)
        else:
            dist_mat += np.square(dis_x_pbc)
    return dist_mat
