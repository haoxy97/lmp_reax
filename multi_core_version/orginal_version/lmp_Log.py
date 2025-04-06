#!/usr/bin/env python3

from curses.ascii import isdigit
import re
import os
import pandas as pd
from pathlib import Path
import numpy as np
from functools import lru_cache
from glob import glob

re_digital = re.compile(r"^[0-9 -\.]+$")


def is_number_string(string):
    return bool(re_digital.match(string))


def log_order(log_name: str):
    """
    log.lammps1 log.lammps2 log.lammps3... log.lammps
    """
    val = 10
    if isdigit(log_name[-1]):
        val = int(log_name[-1])
    return val


class Log:
    def __init__(self, path='./'):
        self.path = path
        if len(glob(os.path.join(path, "log.lammps*"))) > 1:
            self.raw_context = self.read_context_with_restart()
        else:
            self.raw_context = Path(os.path.join(path, "log.lammps")).read_text()
        if 'RESTART' in self.raw_context[:10000]:
            context=re.search(r'([\s\S]+?)write_restart.*\n([\s\S]+)',self.raw_context)
            self.head=context.group(1)
            self.context=context.group(2)
        else:
            context = self.raw_context.split("INIT_DONE")
            self.head = context[0]
            if len(context) > 1:
                self.context = context[1]
            else:
                self.context = ""

    def read_context_with_restart(self):
        """
        log with restart case: there are log.lammps1 log.lammps2 log.lammps3...
        read all of them in order of modified time.
        """
        file_list = glob(os.path.join(self.path, "log.lammps*"))
        file_list.sort(key=log_order)
        return "".join([Path(x).read_text() for x in file_list])

    @property
    @lru_cache(maxsize=None)
    def nvt_run_step(self):
        return int(re.search(r"fix 1 all nvt.*\nrun\s+(\d+)", self.context[:1000]).group(1))

    @property
    @lru_cache(maxsize=None)
    def time_step(self):
        time_step = 1
        re_time_step = re.search(r"timestep\s+([\.0-9]+)", self.head)
        if hasattr(re_time_step, "group"):
            time_step = float(re_time_step.group(1))

        print("path:%-30s,time step:%4.2f" % (self.path, time_step))
        return time_step

    @property
    @lru_cache(maxsize=None)
    def thermo_step(self):
        thermo_step = 1000
        re_thermo_step = re.search(r"thermo\s+([\.0-9]+)", self.head)
        if hasattr(re_thermo_step, "group"):
            thermo_step = int(re_thermo_step.group(1))
        return thermo_step

    def get_performance(self):
        print(re.findall(r'Performance: .*\n', self.raw_context))

    @property
    @lru_cache(maxsize=None)
    def thermo_index(self):

        thermo_index = re.search(r'Mbytes\n(.*)\n', self.context[:10000])
        if thermo_index is not None:
            thermo_index = thermo_index.group(1).split()
        else:
            thermo_index = re.search(r'thermo_style custom(.*)\n',
                                     self.context[:10000])
            if thermo_index is not None:
                thermo_index=thermo_index[0].groups().replace("step",'Step').split()
            else:
                thermo_index = []
        return thermo_index

    @property
    @lru_cache(maxsize=None)
    def thermo_array(self):
        raw_data_list = re.findall(r"Mbytes\n.*\n([\s\S]*?)\nLoop", self.context)
        raw_data_list = sum([item.split("\n")[:-1] for item in raw_data_list], [])  # drop each case last line
        data_array = np.array([item.split() for item in raw_data_list if is_number_string(item)], dtype=float)
        return data_array

    @property
    @lru_cache(maxsize=None)
    def thermo_data(self):
        sample_captivity = int(1e6 / 50 / (self.time_step * self.thermo_step))  # 100 pints per ns
        data_array = self.thermo_array
        data_array = [data_array[i:i + sample_captivity].mean(axis=0)
                      for i in range(0, len(data_array), sample_captivity)]
        data_frame = pd.DataFrame(data_array, columns=self.thermo_index[:len(data_array[0])])
        data_frame.set_index('Step', inplace=True)
        data_frame.index.name = 'Time(ns)'
        if "Density" in data_frame.columns:
            data_frame['Density'] *= 1000  # density unit kg/m3
        data_frame.index *= self.time_step / 1e6
        data_frame.to_csv('%s/log.csv' % self.path, float_format="%.2f")
        return data_frame


if __name__ == '__main__':
    log = Log(path='example/res_1').thermo_data
