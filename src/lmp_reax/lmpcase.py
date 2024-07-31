#!/usr/bin/env python3

"""
stat each single trace of MD into case
"""
import os
from glob import glob
import pickle
from pathlib import Path
from collections import Counter
import re
import numpy as np
from .lmptrace import RxnPath, RxnBondTrace, RxnSpecieTrace
from .lmp_env import OUTPUT_TIME_STEP_NUM, SPECIE_ENLARGE
from . import lmpdatatrace
re_digital = re.compile(r'[0-9]+')


def get_timestep(file_path: str) -> float:
    context = Path(glob(os.path.join(file_path, '*inp'))[0]).read_text()
    return float(re.search('timestep\s+(\S+)', context).group(1))


class LmpCase:
    """
    object of lmp results
    """
    output_time_step = OUTPUT_TIME_STEP_NUM

    def __init__(self, case_folder: str):
        self.case_folder = case_folder
        self.time_step = get_timestep(case_folder)
        self.avg_sample = None
        self.step_list = []
        self.bond_list = []
        self.specie_list = []
        self.rxn_list = []
        self.load_batch_()
        self.volume = self.get_volume()
        self.specie_trace = None

    def get_volume(self):
        # get volume from data file, unit A^3
        context = list((glob(os.path.join(self.case_folder, '*.data'))))
        if len(context) > 1:
            print('Warning, multiple data file, please delete unrelated ones')
        context = Path(context[0]).read_text()
        xyz = re.findall(r'\s*(.*)lo.*\n', context)
        v = 1
        for item in xyz:
            l = item.split()[:2]
            v *= float(l[1]) - float(l[0])
        return abs(v)

    def load_batch_(self, batch_folder='trace'):
        """load all the pickle results into one list by sum method"""
        step_list = []
        bond_list = []
        specie_list = []
        rxn_list = []
        batch_folder = os.path.join(self.case_folder, batch_folder)

        for pickle_file in sorted(glob(os.path.join(batch_folder, '*.pickle')),
                                  key=lambda x: int(os.path.basename(x).split('.')[0])):

            if os.path.getsize(pickle_file) < 1:  # if no result skip
                continue
            trace = pickle.load(Path(pickle_file).open("rb"))
            step_list += trace.step
            bond_list += trace.bond
            specie_list += trace.specie
            rxn_list += trace.rxn

        # average the result each 0.01ns
        self.avg_sample = int(self.output_time_step / (step_list[1] - step_list[0]) / self.time_step)

        iter_index = list(range(0, len(step_list), self.avg_sample))

        # check the last line len, if last sample size is less than the avg_sample, drop it
        if len(step_list) - iter_index[-1] < self.avg_sample:
            iter_index.pop()

        step_list = np.array([step_list[index] for index in iter_index], dtype=float)
        step_list *= self.time_step
        self.step_list = step_list.astype(int).tolist()

        # average the result each 0.01ns
        raw_bond_list = [sum(bond_list[index:index + self.avg_sample], Counter()) for index in iter_index]
        self.bond_list = [Counter(dict([(x, val // self.avg_sample) for x, val in item.items()])) for item in
                          raw_bond_list]

        # average the result each 0.01ns
        sample_specie_list = [sum(specie_list[index:index + self.avg_sample], Counter()) for index in iter_index]
        # NOTE, specie num is enlarge with 1000.
        self.specie_list = [Counter(dict([(x, val * SPECIE_ENLARGE // self.avg_sample) for x, val in item.items()])) for
                            item in
                            sample_specie_list]

        # average the result each 0.01ns
        self.rxn_list = [sum(rxn_list[index:index + self.avg_sample], Counter()) for index in iter_index]


    def plot_all(self):
        rxn_save_path = os.path.join(self.case_folder, "reax_post")
        Path(rxn_save_path).mkdir(exist_ok=True)
        RxnSpecieTrace(data=self.specie_list, index=self.step_list, save_path=rxn_save_path).plot()
        RxnBondTrace(data=self.bond_list, index=self.step_list, save_path=rxn_save_path).plot()
        rxn_path = RxnPath(data=self.rxn_list, index=self.step_list, save_path=rxn_save_path)
        rxn_path.save_csv()
        # for specie in RXN_START_SPECIE:
            # rxn_path.plot_path(start_specie=specie, threshold=PATH_THRESHOLD)
        # for specie in RXN_START_SPECIE + FLUX_SPECIE:
            # rxn_path.plot_flux(specie=specie, threshold=PATH_THRESHOLD)

    def __repr__(self):
        return "LmpCase %s" % self.case_folder
