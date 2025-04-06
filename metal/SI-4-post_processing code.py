#!/usr/bin/env python3

"""
# usage, after finishing the MD simulation, cd {the_work_dir}; python3 lmp_reax2.py;
# the result will shown in `reax_post` folder

# Installation:
Platform capability: tested in Ubuntu 22 and python 3.10

requirements, use pip to install packages: matplotlib, graphviz, netcdf4


You need create a dir and files as below code strcture:

- lmp_reax2.py (main file)
- reax2(folder name)
-- __init__.py (empty file)
-- env.py
-- case.py
-- frame.py
-- block.py
"""

##### start of env.py ####
from dataclasses import dataclass

@dataclass
class EnvFrame:
    type2element_str_dict = {1: "Al", 2: "H", 3: "O", 4: "N"}
    element_stro2typedict = {"Al": 1, "H": 2, "O": 3, "N": 4}
    type2element_id_dict = {1: 13, 2: 1, 3: 8, 4: 7}
    MAX_SMILE_MOLE_NUM = 30
    METAL = True
    METAL_ID = 1
    OXYGEN_ID = 3
    METAL_NAME = 'Al'
    distance_sq_ref = {(1, 1): 2.89, (1, 2): 1.79, (1, 3): 1.97,
                       (2, 2): 0.737, (2, 3): 0.976,
                       (3, 3): 1.188849}
    distance_sq_ref = {x: (y * 1.2) ** 2 for x, y in distance_sq_ref.items()}
    changes = {}
    for x, y in distance_sq_ref.items():
        rev_x = x[1], x[0]
        if distance_sq_ref.get(rev_x, None) is None:
            changes[rev_x] = y
    distance_sq_ref.update(changes)

@dataclass
class EnvBlock:
    CACHE = True
    skip_frame = 1
    worker = 8
    sub_block_size = 100  
    key_reaction = 'HO->#metal#(HO)'
    key_reaction_re = "->".join(key_reaction.split("->")[::-1])


@dataclass
class EnvCase:
    DataSourceNC = True  
    SampleVal = 1
    MinSpecieVal = 0.1
    RXN_START_SPECIE = "H2O"
    SPECIE_ENLARGE = 1e3
    OUTPUT_TIME_STEP_NUM = 1e4  
    RATE_TIME_SECTION = 1  
    FLUX_SPECIE = ('H', "OH", "HO", 'H2O', 'H2')
    PATH_THRESHOLD = [2 ** x for x in range(11)]
    PATH_MAX_NUM = 255
    PATH_MIN_NUM = 3

#####   end of env.py ####

##### start of case.py ####
from itertools import repeat
import cairosvg
from glob import glob
import os
from .block import Block, NcParser, init_frame_env
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import utility
from .env import EnvFrame
import re
from itertools import groupby
import math
from tqdm import tqdm
from multiprocessing import Pool
from lmp_io.atom import Atom
from collections import Counter
digital_re = re.compile(r'\d+')


class Case:
    block: Block

    def __init__(self, work_path: str, skip_frame=1, use_cache=True) -> None:
        self.work_path = work_path
        self.use_cache = use_cache
        self.skip_frame = skip_frame
        NcParser.skip_frame = skip_frame
        self.save_path = os.path.join(work_path, 'reax_post')
        Path(self.save_path).mkdir(exist_ok=True)
        self.load_nc()

    def load_nc(self):

        
        args = list(zip(self.nc_list, repeat(self.nc_list[0]),
                        repeat(self.use_cache)))
        with Pool(8) as p:
            res = list(tqdm(
                p.imap(NcParser(self.skip_frame).parse_packed, args),
                       total=len(self.nc_list),
                       desc='load nc'))
            
            
        if False:
            res = []
            frame_env = init_frame_env(self.nc_list[0])
            for nc in tqdm(self.nc_list, total=len(self.nc_list), desc='load nc'):
                res.append(NcParser().parse(nc, frame_env=frame_env, use_cache=self.use_cache))
        self.block = Block(np.concatenate([x.steps for x in res], axis=0),
                           np.concatenate([x.specie_list for x in res], axis=0))

        

    def dump_specie(self, specie_names=['#METAL#H1O1', '#METAL#H2O1', '#METAL#H', '#METAL#O','H2']):
        "output the specie cords in the atom format, for the visual analysis in the ovito"
        frame_env = init_frame_env(self.nc_list[0])
        Path(self.save_path, 'specie_cords').mkdir(exist_ok=True)
        output_all = [list() for _ in specie_names]
        for nc in tqdm(self.nc_list, total=len(self.nc_list), desc='dump specie'):
            nc_data = NcParser().nc_parser(nc)
            specie_tables = NcParser().parse(nc, frame_env=frame_env, use_cache=True).specie_list
            for specie_i, specie_name in enumerate(specie_names):
                for i, specie_table in enumerate(specie_tables):
                    key_ids = np.where(specie_table == specie_name)
                    if len(key_ids) == 0:
                        continue
                    xyz_array = nc_data[i][2]
                    output_all[specie_i].append(Atom(
                        n_atom=len(xyz_array),
                        box_lengths=nc_data[i][1],
                        id_array=np.arange(1, len(xyz_array)+1, 1),
                        time_step=nc_data[i][0],
                        q_array=np.zeros(len(xyz_array)),
                        v_array=np.zeros_like(xyz_array),
                        xyz_array=xyz_array, type_array=frame_env.type_array, bounds=np.stack(
                            (xyz_array.min(axis=0),
                             xyz_array.max(axis=0)),
                            axis=1)).dump_atom(
                        save_name=os.path.join(self.save_path, '%s.atom' % nc_data[i][0]),
                        atom_ids=key_ids[0], return_str=True))
        for specie_name, output in zip(specie_names, output_all):
            Path(os.path.join(self.save_path, 'specie_cords', '%s.atom' % specie_name)).write_text('\n'.join(output))

    @ property
    @ lru_cache(maxsize=None)
    def nc_list(self, nc_folder='1'):
        return sorted(glob(os.path.join(self.work_path, nc_folder, '*.nc')),
                      key=lambda x: int(os.path.basename(x).split('.')[0]))

    def plot_specie_trace(self, threshold=5):
        metal_name = EnvFrame.METAL_NAME
        utility.set_graph_format()
        Path(os.path.join(self.save_path, 'specie')).mkdir(exist_ok=True)
        x = self.block.steps/1e6
        specie_col = np.unique(self.block.specie_list)
        counter_list = [Counter(x) for x in self.block.specie_list]
        df = np.array([[counter_list[i][specie] for i in range(len(counter_list))] for specie in specie_col])

        
        for i, (key, val) in enumerate(zip(specie_col, df)):
            if metal_name not in key:
                atom_num = max(np.sum([int(x) for x in digital_re.findall(key)]), 1)
                df[i] = val/atom_num

        
        sortidx = np.flip(np.argsort(np.max(df, axis=1)))
        header = "time(ps),"+",".join(['%s' % x for x in specie_col[sortidx]])
        np.savetxt(os.path.join(self.save_path, 'specie.csv'), np.hstack(
            (self.block.steps.reshape(-1, 1), df[sortidx].T)), delimiter=',', fmt='%d', header=header, comments='')

        
        for key, val in tqdm(zip(specie_col, df), total=len(specie_col), desc='plot specie'):
            if val.max() < threshold:
                continue
            plt.figure()
            plt.plot(x, val, label=key)
            plt.xlabel('Time(ns)')
            plt.ylabel('Number')
            plt.title(key)
            plt.savefig(os.path.join(self.save_path, 'specie', '%s.png' % key))
            plt.close()

    def plot_rxn_path(self, threshold=5, max_line=100, start="H2O1"):
        """rxn is the speice path of specie"""
        trace = []
        csv__trace_output = []
        for atom_trace in self.block.specie_list.T:
            
            per_trace = [key for key, group in groupby(atom_trace) if len(list(group)) > 1]
            
            per_trace = [key for key, group in groupby(per_trace)]

            
            if len(per_trace) > 1:
                trace.append(['%s->%s' % x for x in zip(per_trace[:-1], per_trace[1:])])
            csv__trace_output.append(','.join(per_trace)+'\n')
        Path(os.path.join(self.save_path, 'atom_trace.csv')).write_text(''.join(csv__trace_output))
        counter = Counter(sum(trace, []))

        rxns = []
        metal_name = EnvFrame.METAL_NAME
        for key, val in counter.items():
            split_key = key.split('->')
            re_key = "->".join(split_key[::-1])
            if metal_name not in key:
                atom_num = max(np.sum([int(x) for x in digital_re.findall(split_key[0])]),
                               np.sum([int(x) for x in digital_re.findall(split_key[1])]), 1)
            else:
                atom_num = 1
            if re_key in counter:
                re_val = counter[re_key]
                diff = val-re_val
                if diff >= 0:
                    

                    rxns.append((key, diff//atom_num, val//atom_num, re_val//atom_num))
                
                
            else:
                rxns.append((key, val//atom_num, val//atom_num, 0))
        rxns = sorted(rxns, key=lambda x: x[1], reverse=True)

        
        Path(
            os.path.join(self.save_path, 'rxn.csv')).write_text(
            '\n'.join(
                ['rxn,diff,forward,reverse'] +
                ['%s,%d,%d,%d' % x for x in rxns if x[1] > threshold or x[2] > threshold]))

        
        from graphviz import Digraph
        key_trace = {x[0]: x[1] for x in rxns[:min(max_line, len(rxns)-1)]}

        g = Digraph(os.path.join(self.save_path, 'path'), format='svg',
                    node_attr={'fontname': 'arial'},
                    edge_attr={'fontname': 'arial'},
                    graph_attr={'fontname': 'arial'})
        g.attr('node', shape='box')
        g.node(start, start, rank='source')

        edge_width_array = np.array(list(key_trace.values()))
        min_val = edge_width_array.min()
        max_val = edge_width_array.max()
        for rxn, val in key_trace.items():
            r, p = rxn.split('->')
            counter = 0
            
            
            font_color = 'blue%d' % (counter % 5) if counter > 0 else 'blue'
            edge_width = 1 + 4 * math.sqrt((val - min_val) / max_val)
            
            g.edge(r, p, label='%d' % val, fontcolor=font_color, color='gray',
                   penwidth='%.2f' % edge_width, arrowsize='1', arrowhead='empty', fontsize='12')
            counter += 1
        render_svg_filepath = g.render()
        render_svg_filepath = os.path.abspath(render_svg_filepath)
        cairosvg.svg2png(url=render_svg_filepath, write_to=render_svg_filepath.replace('.svg', '.png'))

    def plot(self):
        self.plot_rxn_path()
        self.plot_specie_trace()

    def dump_all(self):
        self.dump_specie()

def remove_cache(path):
    cache_list=glob(os.path.join(path, '1', '*.npz'))
    env_file=os.path.join(path, '1', 'env.npz')
    if env_file in cache_list:
        cache_list.remove(os.path.join(path, '1', 'env.npz'))
    cache_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
    cache_step_length_list=[first_last_difference(Block.load(x).steps) for x in cache_list]
    most_step=Counter(cache_step_length_list).most_common(1)[0][0]
    for cache, step_length in zip(cache_list, cache_step_length_list):
        if step_length != most_step:
            Path(cache).unlink()
            print('remove %s' % cache)                
    print('remove cache done')

def first_last_difference(x):
    return np.subtract(x[-1], x[0])
#####   end of case.py ####

##### start of frame.py ####
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
        
        atom_neighbor_list = [list() for _ in range(self.env.n_atoms)]
        
        id_array = np.arange(self.env.n_atoms)
        for i, item in enumerate(squareform(distance_sq_mat(cord, box) < self.env.dis_sq_ref)):
            item = id_array[item]
            if len(item) > 0:
                atom_neighbor_list[i] = item
                
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
        for center_id, neighbor_idx in enumerate(neighbor_list):
            if type_array[center_id] == metal_id:
                name = [metal_name]
                metal_neighbor_count = 0
                non_metal_neighbor = []
                non_metal_name = []
                for idx in neighbor_idx:
                    if type_array[idx] == metal_id:
                        pass
                    else:
                        non_metal_neighbor.append(idx)

                if metal_neighbor_count > 0:
                    name.append("(%s)%d" % (metal_name, metal_neighbor_count))

                for non_metal_id in non_metal_neighbor:
                    if specie_id_list[non_metal_id] is None:
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
                                non_metal_sub = np.array(sub_sub_neighbor)[sub_sub_type != 1]
                                sub_stack = np.concatenate([sub_neighbor, non_metal_sub, [non_metal_id]])

                                if len(non_metal_sub) == 1:
                                    sub_sub_sub_neighbor = neighbor_list[non_metal_sub[0]].tolist()
                                else:
                                    sub_sub_sub_neighbor = list(
                                        set(np.concatenate([neighbor_list[x] for x in non_metal_sub])))
                                
                                sub_sub_sub_neighbor = [x for x in sub_sub_sub_neighbor if x not in sub_stack]
                                sub_sub_sub_type = type_array[sub_sub_sub_neighbor]
                                if np.sum(sub_sub_sub_type) != metal_id * len(sub_sub_sub_neighbor):
                                    print('warning: 3 layer not all metal')
                            sub_name = self.parse_specie(sub_stack)
                            non_metal_name.append("%s" % sub_name)
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
            dist_mat = None
    for dim in range(len(box_length)):
        dis_x = np.float32(pdist(cords[:, dim][:, np.newaxis]))
        dis_x_pbc = np.minimum(box_length[dim] - dis_x, dis_x)
        if dist_mat is None:
            dist_mat = np.square(dis_x_pbc)
        else:
            dist_mat += np.square(dis_x_pbc)
    return dist_mat

#####   end of frame.py ####

##### start of block.py ####
import numpy as np
from dataclasses import dataclass
from netCDF4 import Dataset
from .env import EnvFrame
from .frame import FrameEnv, Frame
from pathlib import Path
import os


def init_frame_env(nc_file: str) -> FrameEnv:
    cache = os.path.join(os.path.dirname(nc_file), 'env.npz')
    if Path(cache).exists():
        npz = np.load(cache)
        n_atom = int(npz['n_atoms'])
        type_array = npz['type_array']
        element_array = npz['element_array']
        mat = npz['dis_sq_ref']
        return FrameEnv(n_atom, type_array, element_array, mat)

    with Dataset(nc_file, mode='r', format="NETCDF4") as nc:
        type_array = np.array(nc.variables['type'][0], dtype=int)
    n_atom = len(type_array)
    element_array = np.array([EnvFrame.type2element_str_dict[x] for x in type_array], dtype=str)
    mat = []
    for i in range(n_atom - 1):
        type_x = type_array[i]
        mat += [EnvFrame.distance_sq_ref[(type_x, type_array[j])]
                for j in range(i + 1, n_atom)]
    mat = np.array(mat)
    np.savez_compressed(cache, n_atoms=n_atom, type_array=type_array, element_array=element_array, dis_sq_ref=mat)
        return FrameEnv(n_atom, type_array, element_array, mat)


@dataclass
class Block:
    steps: np.ndarray
    specie_list: np.ndarray

    def __add__(self, other: "Block"):
        self.steps = np.concatenate((self.steps, other.steps), axis=0)
        self.specie_list = np.concatenate((self.specie_list, other.specie_list), axis=0)
        return self

    def save(self, file_path):
        np.savez_compressed(file_path, steps=self.steps, specie_list=self.specie_list)

    @classmethod
    def load(cls, file_path):
        return cls(**np.load(file_path))

    def save_csv(self, file_path):
        header = ",".join(['%d' % x for x in self.steps])
        np.savetxt(file_path, np.array(self.specie_list).T,
                   delimiter=',', fmt='%s', header=header, comments='#time(ps)/atom_ids# ')


class NcParser:
    skip_frame = 1

    def __init__(self, skip_frame=None):
        if skip_frame is not None:
            self.skip_frame = skip_frame

    def nc_parser(self, nc_file: str):
        with Dataset(nc_file, mode='r', format="NETCDF4") as nc:
            step_traj = np.array(nc.variables['time'][:-1:self.skip_frame]).astype(int)
            cord_traj = np.array(nc.variables['coordinates'][:-1:self.skip_frame])
            box_length_traj = np.array(nc.variables['cell_lengths'][:-1:self.skip_frame])
        return list(zip(step_traj, box_length_traj, cord_traj))

    def parse(self, nc_file: str, frame_env: FrameEnv = None, use_cache=True):
        cache = nc_file+'.npz'
        if Path(cache).exists() and use_cache:
            return Block.load(cache)
        if isinstance(frame_env, str):
                        frame_env = init_frame_env(frame_env)
        steps = []
        speice_list = []
        for step, box, cord in self.nc_parser(nc_file):
            frame = Frame(step, cord, box, frame_env)
            steps.append(frame.step)
            speice_list.append(frame.specie_id_list)
        block = Block(np.array(steps), np.array(speice_list))
        block.save(file_path=nc_file+'.npz')
                return block

    def parse_packed(self, args):
        nc_file, frame_env, use_cache = args
        cache = nc_file+'.npz'
        if Path(cache).exists() and use_cache:
            return Block.load(cache)
        if isinstance(frame_env, str):
                        frame_env = init_frame_env(frame_env)
        steps = []
        speice_list = []
        for step, box, cord in self.nc_parser(nc_file):
            frame = Frame(step, cord, box, frame_env)
            steps.append(frame.step)
            speice_list.append(frame.specie_id_list)
        block = Block(np.array(steps), np.array(speice_list))
        block.save(file_path=nc_file+'.npz')
                return block

#####   end of block.py ####

##### start of lmp_reax2.py ####
#!/usr/bin/env python3
from reax2.case import Case,remove_cache
from utility import ArgvParser


def run(path=None):
    argv = ArgvParser(default_path='.')
    argv.parser.add_argument('-sf', '--skipFrame', help='[10]', type=int, default=10)
    argv.parser.add_argument('-c', '--cache', help='use cache, [True]', default=True, action='store_false')
    argv.parser.add_argument('-r', '--remove', help='remove uncomplete cache[False]', default=False, action='store_true')
    if argv.remove:
        remove_cache(argv.path)
        exit()
    if path is not None:
        argv.path = path
    else:
        case = Case(argv.path, skip_frame=argv.skipFrame, use_cache=argv.cache)
        case.plot()
        case.dump_all()


if __name__ == '__main__':
    run()

#####   end of lmp_reax2.py ####