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

        # the val need correctlated since it based on specie list with multiple atom
        for i, (key, val) in enumerate(zip(specie_col, df)):
            if metal_name not in key:
                atom_num = max(np.sum([int(x) for x in digital_re.findall(key)]), 1)
                df[i] = val/atom_num

        # now save csv
        sortidx = np.flip(np.argsort(np.max(df, axis=1)))
        header = "time(ps),"+",".join(['%s' % x for x in specie_col[sortidx]])
        np.savetxt(os.path.join(self.save_path, 'specie.csv'), np.hstack(
            (self.block.steps.reshape(-1, 1), df[sortidx].T)), delimiter=',', fmt='%d', header=header, comments='')

        # now plot
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
            # use len(list(group)) > 1 to filter osculation
            per_trace = [key for key, group in groupby(atom_trace) if len(list(group)) > 1]
            # do the sam group again for the osculation clear
            per_trace = [key for key, group in groupby(per_trace)]

            # per_trace = [(key,len(list(group))) for key, group in groupby(atom_trace)]
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
                    # the val need correctlated since it based on specie list with multiple atom

                    rxns.append((key, diff//atom_num, val//atom_num, re_val//atom_num))
                # else:
                #     rxns.append((re_key, -diff, re_val, val))
            else:
                rxns.append((key, val//atom_num, val//atom_num, 0))
        rxns = sorted(rxns, key=lambda x: x[1], reverse=True)

        # now merge the revserse
        Path(
            os.path.join(self.save_path, 'rxn.csv')).write_text(
            '\n'.join(
                ['rxn,diff,forward,reverse'] +
                ['%s,%d,%d,%d' % x for x in rxns if x[1] > threshold or x[2] > threshold]))

        # now plot it
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
            # if r not in linked_specie_list or p not in linked_specie_list:
            #     continue
            font_color = 'blue%d' % (counter % 5) if counter > 0 else 'blue'
            edge_width = 1 + 4 * math.sqrt((val - min_val) / max_val)
            # edge_text = reactant.replace("+", '').replace('r', '')
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