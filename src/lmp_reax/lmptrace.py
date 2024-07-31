#!/usr/bin/env python3


import os
from collections import Counter
import pandas as pd
import re
import matplotlib.pyplot as plt
from functools import lru_cache
from collections import OrderedDict
from .lmp_env import type2element_str_dict
from pathlib import Path
import subprocess
from .lmp_env import SPECIE_ENLARGE
from typing import List, Union
from tqdm import tqdm
from .lmp_env import FLUX_SPECIE, METAL, PATH_MAX_NUM, PATH_MIN_NUM


element_weight = {"N": 400, 'C': 100, 'H': 10, 'O': 1}
re_element_count = re.compile(r'[A-Z]\d*')


def weight_specie(specie: str):
    """
    note, element only can use one digital string
    """
    weight = 0
    specie = specie.split("__")[0]
    if "--" not in specie:
        specie = re_element_count.findall(specie)
        for item in specie:
            if len(item) == 1:
                weight += element_weight[item]
            else:
                weight += element_weight[item[0]] * int(item[1:])
    return weight


def format_specie_inner_bond(specie_or_bond: str):
    """note  'HO' is convert to 'OH'"""
    if "--" not in specie_or_bond:
        if specie_or_bond == "HO":
            return "OH"
        return specie_or_bond
    a, b = specie_or_bond.split("--")
    if element_weight[a] < element_weight[b]:
        return "%s--%s" % (b, a)
    else:
        return specie_or_bond


def format_reaction(reaction: str):
    """
    eliminate the distinguish of A+B and B+A
    by weight the species in both R and P side,
    """
    r, p = reaction.split("->")
    if "+" in r:
        r = r.split("+")
        r = [format_specie_inner_bond(x) for x in r]
        r.sort(key=weight_specie, reverse=True)
        r = "+".join(r)
    if "+" in p:
        p = p.split("+")
        p = [format_specie_inner_bond(x) for x in p]
        p.sort(key=weight_specie, reverse=True)
        p = "+".join(p)
    return "%s->%s" % (r, p)


def smi_2_svg(smi_string: Union[str, list]) -> subprocess.CompletedProcess:
    if isinstance(smi_string, str):
        smi_string = '-:"%s"' % smi_string
    else:
        smi_string = " ".join(['-:"%s"' % x for x in smi_string])
    # res = subprocess.run("obabel %s -o svg -xi -xj " % smi_string,
    # obabel -xi # add index to atom
    # obabel -xj # add no javascript in svg output
    res = subprocess.run("obabel %s -o svg -xj " % smi_string,
                         capture_output=True, text=True, shell=True)
    return res


class RxnBondTrace:
    def __init__(self, data, index, save_path='./'):
        frame = pd.DataFrame(data=data, index=index).fillna(0).astype(int)
        frame.index.name = 'time(ps)'
        frame.to_csv(os.path.join(save_path, 'bond_trace.csv'))
        self.frame = frame
        self.save_path = save_path

    def plot(self):

        frame = self.frame
        time_index = frame.index / 1e6
        colors = 'tab:blue', 'tab:orange', 'tab:gray', 'tab:pink', 'lightgreen', 'dimgrey','cyan','black'
        plot_bond_types = ["11", "12", "22", "23"]
        plot_bond_names = ["-".join([type2element_str_dict[int(x)] for x in item]) for item in plot_bond_types]

        # fill empty bonds to keep colors unchanged
        for item in plot_bond_types:
            if frame.get(item) is None:
                frame[item] = pd.Series(range(10))
        fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.75)

        twin1 = ax.twinx()
        twin2 = ax.twinx()
        twin3 = ax.twinx()

        # move twins ax away
        twin2.spines['right'].set_position(("axes", 1.25))
        twin3.spines['right'].set_position(("axes", 1.5))

        p1, = ax.plot(time_index, frame[plot_bond_types[0]].values, c=colors[0], label=plot_bond_names[0])
        p2, = twin1.plot(time_index, frame[plot_bond_types[1]].values, c=colors[1], label=plot_bond_names[1])
        p3, = twin2.plot(time_index, frame[plot_bond_types[2]].values, c=colors[2], label=plot_bond_names[2])
        p4, = twin3.plot(time_index, frame[plot_bond_types[3]].values, c=colors[3], label=plot_bond_names[3])

        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(plot_bond_names[0])
        twin1.set_ylabel(plot_bond_names[1])
        twin2.set_ylabel(plot_bond_names[2])
        twin3.set_ylabel(plot_bond_names[3])

        ax.yaxis.label.set_color(p1.get_color())
        twin1.yaxis.label.set_color(p2.get_color())
        twin2.yaxis.label.set_color(p3.get_color())
        twin3.yaxis.label.set_color(p4.get_color())

        tkw = dict(size=4, width=1.5)
        ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
        twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        twin3.tick_params(axis='y', colors=p4.get_color(), **tkw)
        ax.tick_params(axis='x', **tkw)

        ax.legend(handles=[p1, p2, p3, p4])
        plt.savefig(os.path.join(self.save_path, 'bond.png'))
        plt.close()


class RxnSpecieTrace:
    def __init__(self, data, index, save_path='./'):
        frame = pd.DataFrame(data=data, index=index).fillna(0).astype(int)
        frame.index.name = 'time(ps)'
        self.save_path = save_path
        # trace is too large as cache
        # frame.to_csv(os.path.join(save_path, 'specie_trace.csv'))
        self.frame = frame
        self.specie_dict = None
        self.simplify_frame_()

    def simplify_frame_(self):
        """
        simplify the species by merge those with same same formula,
        and plot the smi svg, also add the fraction of smi,
        save the total smi dict as csv.
        the fraction = 100 * max( current smi specie in trace ) / max( species with same formula in trace)
        """
        re_sub_smi = re.compile(r'#.*')  # remove smi
        re_sub_formula = re.compile(r".*#")  # remove formula

        # use columns to merge species with same formula
        specie_dict = {}
        for raw_specie in self.frame.columns:
            specie = re_sub_smi.sub('', raw_specie)
            specie_dict[specie] = specie_dict.get(specie, []) + [raw_specie]

        # merge specie_frame to simple frame
        simple_frame_data_list = []
        for key, val in specie_dict.items():
            simple_frame_data_list.append(self.frame[val].sum(1).values)
        simple_frame = pd.DataFrame(data=simple_frame_data_list, columns=self.frame.index, index=specie_dict.keys()).T

        raw_specie_dict_context = ["specie,total_smiles,smiles_fraction,smi_string\n"]
        raw_specie_max_dict = self.frame.max().to_dict()
        simple_specie_max_dict = simple_frame.max().to_dict()
        self.frame = simple_frame
        self.specie_dict = specie_dict
        if self.save_path is None:
            return None

        svg_folder = os.path.join(self.save_path, 'svg')
        if not os.path.exists(svg_folder):
            os.mkdir(svg_folder)
        for key_val in tqdm(specie_dict.items(), total=len(specie_dict.keys()), desc='plot smi'):
            key, val = key_val
            if "#" not in "".join(val):  # "#" mark is for specie with smi, if not smi, skip it
                continue
            raw_specie_fraction_list = [
                (int(raw_specie_max_dict[x] * 100 / simple_specie_max_dict[key]), re_sub_formula.sub("", x))
                for x in val]
            raw_specie_fraction_list.sort(key=lambda x: x[0], reverse=True)
            raw_specie_dict_context.append(
                "%s,%d,%s,%s\n" % (
                    key, len(raw_specie_fraction_list),
                    " ".join(["%d" % x for x, y in raw_specie_fraction_list]),
                    " ".join(["%s" % y for x, y in raw_specie_fraction_list])))

            svg_filepath = (os.path.join(svg_folder, "%s#%d.svg" % (key, len(raw_specie_fraction_list))))
            if os.path.exists(svg_filepath):
                continue
            # if len(raw_specie_fraction_list) < 2:  # for molecular with no multi structure, ignore svg plot
            #     continue
            # direct use open babel to parse smi to svg
            res = smi_2_svg([y for x, y in raw_specie_fraction_list])
            if len(res.stdout) < 10:
                continue

            # add the smi fraction as a title in the svg
            svg_context = res.stdout.split('\n')
            if len(raw_specie_fraction_list) > 1:
                smi_val_list = []
                for smi_val in raw_specie_fraction_list:
                    smi_val = smi_val[0]
                    smi_val_string = '<text x="10" y="15" stroke-width="0" font-weight="bold" font-size="18" >%d</text>' \
                                     % smi_val
                    smi_val_list.append(smi_val_string)
                smi_counter = 0
                for index, line in enumerate(svg_context):
                    if smi_counter >= len(smi_val_list):
                        break
                    if line == '</svg>':
                        svg_context[index] = "%s\n</svg>" % (smi_val_list[smi_counter])
                        smi_counter += 1

            Path(svg_filepath).write_text("\n".join(svg_context))

        Path(os.path.join(self.save_path), "specie_smi_map.csv").write_text("".join(raw_specie_dict_context))

    def plot(self, minimum_val=5 * SPECIE_ENLARGE):
        """plot key specie trace"""
        frame = self.frame
        save_path = os.path.join(self.save_path, 'specie')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        time_index = frame.index / 1e6
        key_specie_list = [key for key, item in frame.max().to_dict().items()
                           if item > minimum_val]
        key_specie_list = set(key_specie_list + FLUX_SPECIE)
        final_specie_list = []
        for specie in key_specie_list:
            if frame.get(specie) is None:
                continue
            final_specie_list.append(specie)
            plt.figure()
            plt.plot(time_index, frame[specie].values / SPECIE_ENLARGE)
            plt.xlabel('Time (ns)')
            plt.ylabel('%s Counts' % specie)
            plt.title(specie)
            plt.savefig('%s/%s.png' % (save_path, specie))
            plt.close()
        save_frame = frame[sorted(final_specie_list)]
        save_frame /= SPECIE_ENLARGE
        save_frame.to_csv(os.path.join(self.save_path, 'specie_trace.csv'))


class RxnPath:
    def __init__(self, data: List[Counter], index=None, average_val=1, save_path='./'):
        self.save_path = save_path
        data = self.data_remove_smi(data)
        frame = self.init_frame(data, index)
        self.frame = frame
        self.columns = frame.columns
        self.average_val = average_val

    @property
    @lru_cache(maxsize=None)
    def smi_dict(self):
        """read smi dict from parent folder"""
        smi_path = os.path.join(os.path.dirname(self.save_path), 'specie_smi_map.csv')
        smi_dict = dict()
        if not os.path.exists(smi_path):
            return smi_dict
        context = Path(smi_path).read_text().split('\n')
        for line in context[1:-1]:
            item = line.split(',')
            smi_dict[item[0]] = item[-1].split(" ")[0]
        return smi_dict

    @staticmethod
    def init_frame(data, index):
        """construct frame to dataframe, and merge A+B->C and B+A->C"""
        frame = pd.DataFrame(data=data, index=index).fillna(0).astype(int)
        # merge A+B->C and B+A->C, by use dataframe.groupby function
        frame.columns = list(map(format_reaction, frame.columns))
        frame = frame.T.groupby(frame.columns).sum().T
        frame.index.name = 'time(ps)'
        # raw rxn trace is too large, slow to dump as csv
        # frame.to_csv(os.path.join(self.save_path, 'rxn_trace.csv'))
        return frame

    @staticmethod
    def data_remove_smi(data: List[Counter]):
        simple_data = []
        re_remove_smi = re.compile(r'#.*')
        for counter in data:
            data_counter = Counter()
            for rxn_string, val in counter.items():

                R, P = rxn_string.split("->")
                if "+" in R:
                    R = "+".join([re_remove_smi.sub('', x) for x in R.split("+")])
                else:
                    R = re_remove_smi.sub('', R)
                if "+" in P:
                    P = "+".join([re_remove_smi.sub('', x) for x in P.split("+")])
                else:
                    P = re_remove_smi.sub('', P)
                simple_rxn_string = "%s->%s" % (R, P)

                data_counter[simple_rxn_string] = data_counter.get(simple_rxn_string, 0) + val
            simple_data.append(data_counter)
        return simple_data

    @property
    @lru_cache(maxsize=None)
    def path_dict(self):
        path_series = self.frame.sum().sort_values(ascending=False)
        path_series.index.name = 'reaction'
        path_series.name = 'count'
        p_dict = path_series.to_dict()
        pd.Series(p_dict, name='raw_count').to_csv(os.path.join(self.save_path, 'rxn_path.csv'))
        return p_dict

    @property
    @lru_cache(maxsize=None)
    def summed_path_dict(self) -> OrderedDict:
        raw_rxn_path_list = []
        path_dict = self.path_dict
        for reaction in path_dict.keys():
            rev_reaction = self.reverse_reaction_str(reaction)
            raw_rxn_path_list.append((reaction, path_dict[reaction] - path_dict.get(rev_reaction, 0)))

        raw_rxn_path_list.sort(key=lambda x: abs(x[1]), reverse=True)
        rxn_path_dict = OrderedDict()

        # if path val<0, reverse reaction
        for key, val in raw_rxn_path_list:
            if val == 0:
                continue
            if val < 0:
                rxn_path_dict[self.reverse_reaction_str(key)] = -val // self.average_val
            else:
                rxn_path_dict[key] = val // self.average_val
        pd.Series(rxn_path_dict, name='sum_count').to_csv(os.path.join(self.save_path, 'sum_path.csv'))
        return rxn_path_dict

    def save_csv(self):
        self.summed_path_dict
        self.path_dict
    

    @staticmethod
    def reverse_reaction_str(reaction: str):
        """A+B->C+D to C+D->A+B"""
        r, p = reaction.split("->")
        return "%s->%s" % (p, r)

    def __repr__(self):
        return "RxnPath from %d-%d" % (self.frame.index[0], self.frame.index[-1])
