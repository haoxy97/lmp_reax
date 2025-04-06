#!/usr/bin/env python3

"""
stat each single trace of MD into case
"""
import pprint
import os
from glob import glob
import pickle
from pathlib import Path
from collections import Counter
import pandas as pd
import re
from lmp_Log import Log
import numpy as np
import matplotlib.pyplot as plt
import utility
from functools import lru_cache
from typing import List
# from lmptrace import RxnPath, RxnBondTrace, RxnSpecieTrace
from lmp_env import type2element_str_dict, OUTPUT_TIME_STEP_NUM, SPECIE_ENLARGE, RXN_START_SPECIE, FLUX_SPECIE, \
    PATH_THRESHOLD, RATE_TIME_SECTION, METAL
from operator import add
from lmp_env import METAL_NAME

utility.set_graph_format()
re_digital = re.compile(r'[0-9]+')


class LmpCase:
    """
    object of lmp results
    """
    output_time_step = OUTPUT_TIME_STEP_NUM

    def __init__(self, case_folder: str):
        self.case_folder = case_folder
        self.time_step = Log(case_folder).time_step
        self.avg_sample = None
        self.step_list = []
        self.bond_list = []
        self.specie_list = []
        self.rxn_list = []
        self.load_batch_()
        self.volume = self.get_volume()
        # metal no rate cal
        if not METAL[0]:
            self.rate_frame = self.cal_rate()
        self.to_pickle()
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

    def to_pickle(self, save_name=None):
        if save_name is None:
            save_name = "lmp_case"
        pickle.dump(self, Path(os.path.join(self.case_folder, '%s.pickle' % save_name)).open('wb'))

    def load_batch_(self, batch_folder='trace'):
        """load all the pickle results into one list by sum method"""
        step_list = []
        bond_list = []
        specie_list = []
        rxn_list = []
        batch_folder = os.path.join(self.case_folder, batch_folder)

        for pickle_file in sorted(glob(os.path.join(batch_folder, '*.pickle')),
                                  key=lambda x: int(re_digital.search(os.path.basename(x)).group())):

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

    def get_key_specie_step(self):
        """find start specie 25%/50%/75% consumption step,"""
        start_specie_trace = self.specie_trace.frame[RXN_START_SPECIE].to_numpy()
        consume_list = [(-1, -1)]
        for consume_ratio in 0.25, 0.5, 0.75, 0.90, 0.95:
            consume_index = np.where(start_specie_trace < start_specie_trace[0] * (1 - consume_ratio))[0]
            if len(consume_index) > 1:
                consume_list.append((consume_ratio, consume_index[0]))
        return consume_list

    def time_step_index(self, interval=0.1) -> list:
        """get the index of each interval unit: ns"""
        interval = interval * 1e6
        step_array = np.array(self.step_list)
        step_interval = int(np.ceil(interval / (step_array[1] - step_array[0])))
        if step_interval > len(step_array):
            return []
        key_index_list = list(range(len(step_array)))
        return key_index_list[::step_interval]

    def plot_all(self):

        bond_trace = RxnBondTrace(data=self.bond_list, index=self.step_list, save_path=self.case_folder)
        self.specie_trace = RxnSpecieTrace(data=self.specie_list, index=self.step_list, save_path=self.case_folder)
        bond_trace.plot()
        self.specie_trace.plot()

        if METAL[0]:
            interval = 0.1
            rxn_save_path = os.path.join(self.case_folder, "rxn_metal")
            key_index_list = self.time_step_index(interval=interval)

            if not os.path.exists(rxn_save_path):
                os.mkdir(rxn_save_path)

            rxn_path = RxnPath(data=self.rxn_list, index=self.step_list, save_path=rxn_save_path)

            if METAL_NAME[0] in RXN_START_SPECIE[0]:
                start_specie=RXN_START_SPECIE[0]
            else:
                start_specie=METAL_NAME[0]
            for threshold in PATH_THRESHOLD:
                rxn_path.plot_path(start_specie=start_specie, threshold=threshold)

            rxn_save_path = os.path.join(self.case_folder, "rxn_metal", 'interval')
            if not os.path.exists(rxn_save_path):
                os.mkdir(rxn_save_path)
            rxn_time_list = []
            sum_rxn_time_list = []
            rxn_time_index = []

            for index_a, index_b in zip(key_index_list[0:-1], key_index_list[1:]):
                save_name = '%.1f-%.1f' % (self.step_list[index_a] / 1e6, self.step_list[index_b] / 1e6)
                rxn_time_index.append(save_name)
                rxn_save_path = os.path.join(self.case_folder, "rxn_metal", 'interval', save_name)
                if not os.path.exists(rxn_save_path):
                    os.mkdir(rxn_save_path)
                rxn_path = RxnPath(data=self.rxn_list[index_a:index_b], index=self.step_list[index_a:index_b],
                                   save_path=rxn_save_path)
                rxn_time_list.append(rxn_path.path_dict)
                sum_rxn_time_list.append(rxn_path.summed_path_dict)

                for threshold in PATH_THRESHOLD:
                    rxn_path.plot_path(start_specie=start_specie, threshold=threshold)

            rxn_interval_df = pd.DataFrame(data=rxn_time_list, index=rxn_time_index).fillna(0).T.to_csv(
                os.path.join(self.case_folder, "rxn_metal", 'rxn_interval.csv'))
            sum_interval_df = pd.DataFrame(data=sum_rxn_time_list, index=rxn_time_index).fillna(0).T.to_csv(
                os.path.join(self.case_folder, "rxn_metal", 'sum_interval.csv'))
            return None

        for consume_ratio, consume_index in self.get_key_specie_step():
            rxn_save_path = os.path.join(self.case_folder, "rxn_consume_%d" % (consume_ratio * 100))
            if not os.path.exists(rxn_save_path):
                os.mkdir(rxn_save_path)
            rxn_path = RxnPath(data=self.rxn_list[:consume_index], index=self.step_list[:consume_index],
                               save_path=rxn_save_path)
            for specie in RXN_START_SPECIE:
                for threshold in PATH_THRESHOLD:
                    rxn_path.plot_path(start_specie=specie, threshold=threshold)

            for specie in RXN_START_SPECIE + FLUX_SPECIE:
                for threshold in PATH_THRESHOLD:
                    rxn_path.plot_flux(specie=specie, threshold=threshold)

    def __repr__(self):
        return "LmpCase %s" % self.case_folder


    def cal_rate(self):
        sample_num = int(1e6 * RATE_TIME_SECTION / OUTPUT_TIME_STEP_NUM)
        rate_record_dict = {}
        rxn_val_record_dict = {}
        specie_record_dict = {}
        rate_case = len(self.step_list) // sample_num  # value for indicate the case numbers, 2ns 2 cases; 3ns 3 cases
        for i in range(rate_case):
            specie_counter = RxnSpecieTrace(data=self.specie_list[i * sample_num:(i + 1) * sample_num],
                                            index=self.step_list[i * sample_num:(i + 1) * sample_num],
                                            save_path=None).frame.sum().to_dict()
            rate_counter = RxnPath(data=self.rxn_list[i * sample_num:(i + 1) * sample_num],
                                index=self.step_list[i * sample_num:(i + 1) * sample_num],
                                save_path=None).frame.sum().to_dict()
            # fix OH naming error
            OH = specie_counter.get('HO')
            if OH is not None:
                specie_counter['OH'] = specie_counter.get('HO')

            for specie, val in specie_counter.items():
                if specie_record_dict.get(specie) is None:
                    specie_record_dict[specie] = [0] * rate_case
                specie_record_dict[specie][i] = val / sample_num / SPECIE_ENLARGE
            for rxn, val in rate_counter.items():
                if "--" in rxn:  # ignore inner bond reaction
                    continue
                reactants = rxn.split('->')[0].split('+')
                k = 0.0
                if len(reactants) == 1:
                    # 1st order, unit s-1
                    k = (1e9 * val) / (specie_counter[reactants[0]] / sample_num / SPECIE_ENLARGE)
                if len(reactants) == 2:
                    # 2st order, unit mol cm3 s-1
                    k = 6.02214 * self.volume / 10.0 * \
                        (1e9 * val) / (specie_counter[reactants[0]] / sample_num / SPECIE_ENLARGE) / \
                        (specie_counter[reactants[1]] / sample_num / SPECIE_ENLARGE)
                if len(reactants) == 3:
                    pass
                if rate_record_dict.get(rxn) is None:
                    rate_record_dict[rxn] = [0.] * rate_case
                    rxn_val_record_dict[rxn] = [0] * rate_case

                rate_record_dict[rxn][i] = k
                rxn_val_record_dict[rxn][i] = val
        rate_frame = pd.DataFrame(rate_record_dict).T
        rate_frame['average'] = rate_frame.mean(1)
        rate_frame.sort_values(by='average', ascending=False, inplace=True)
        rate_frame.index.name = 'reaction'
        rate_frame.to_csv(os.path.join(self.case_folder, 'rate.csv'), float_format='%.2e')

        specie_frame = pd.DataFrame(specie_record_dict).T
        specie_frame['average'] = specie_frame.mean(1)
        specie_frame.sort_values(by='average', ascending=False, inplace=True)
        specie_frame.index.name = 'specie'
        specie_frame.to_csv(os.path.join(self.case_folder, 'specie.csv'))

        rxn_frame = pd.DataFrame(rxn_val_record_dict).T
        rxn_frame['average'] = rxn_frame.mean(1)
        rxn_frame.sort_values(by='average', ascending=False, inplace=True)
        rxn_frame.index.name = 'reaction'
        rxn_frame.to_csv(os.path.join(self.case_folder, 'rxn_val.csv'))

        return rate_frame


class LmpCaseCompare:
    def __init__(self, work_folder: str):
        self.work_folder = work_folder
        self.trim_case_len_()

    @property
    @lru_cache(maxsize=None)
    def case_path_list(self) -> List[str]:
        folder_list = [folder for folder in list(os.listdir(self.work_folder))
                       if os.path.exists(os.path.join(self.work_folder, folder, 'lmp_case.pickle'))]
        folder_list.sort()
        return folder_list

    @property
    @lru_cache(maxsize=None)
    def case_list(self) -> List[LmpCase]:
        """list of LmpCase object"""
        case_list = list(
            map(lambda x: pickle.load(Path(os.path.join(self.work_folder, x, 'lmp_case.pickle')).open('rb'))
                , self.case_path_list))
        return case_list

    def trim_case_len_(self):
        """align the case length"""
        minima_len = min([len(item.step_list) for item in self.case_list])
        for case in self.case_list:
            del case.step_list[minima_len:]
            del case.bond_list[minima_len:]
            del case.specie_list[minima_len:]
            del case.rxn_list[minima_len:]
        return minima_len

    def plot_bond(self):
        save_path = os.path.join(self.work_folder, 'compare_bond')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        frame_list = list(map(lambda x: pd.DataFrame(x.bond_list).fillna(0), self.case_list))
        time_index = np.array(self.case_list[0].step_list) / 1e6
        colors = utility.colors
        plot_bond_types = ["11", "12", "22", "23"]
        plot_bond_names = ["-".join([type2element_str_dict[int(x)] for x in item]) for item in plot_bond_types]

        for index, plot_bond_type in enumerate(plot_bond_types):
            compare_frame_dict = dict()
            plt.figure()
            for case_index, frame in enumerate(frame_list):
                if frame.get(plot_bond_type) is None:
                    continue
                plt.plot(time_index, frame[plot_bond_types[index]], c=colors[case_index % len(colors)])
                compare_frame_dict[self.case_path_list[case_index]] = frame[plot_bond_types[index]]

            pd.DataFrame.from_dict(compare_frame_dict).to_csv(os.path.join(save_path,
                                                                           '%s.csv' % plot_bond_names[index]))

            plt.legend(self.case_path_list)
            plt.xlabel('Time (ns)')
            plt.ylabel("%s Count" % plot_bond_names[index])
            plt.savefig(os.path.join(save_path, '%s.png' % plot_bond_names[index]))
            plt.close()

    def plot_specie(self, minimum_val=10):
        save_path = os.path.join(self.work_folder, 'compare_specie')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        frame_list = list(map(lambda x: x.specie_trace.frame, self.case_list))
        time_index = np.array(self.case_list[0].step_list) / 1e6
        colors = utility.colors
        key_specie_list = []
        for frame in frame_list:
            key_specie_list += [key for key, item in frame.max().to_dict().items()
                                if item > minimum_val]
        for specie in set(key_specie_list):
            compare_frame_dict = dict()
            plt.figure()
            for case_index, frame in enumerate(frame_list):
                if frame.get(specie) is None:
                    continue
                plt.plot(time_index, frame[specie], c=colors[case_index % len(colors)])
                compare_frame_dict[self.case_path_list[case_index]] = frame[specie]
            pd.DataFrame.from_dict(compare_frame_dict).to_csv(os.path.join(save_path, '%s.csv' % specie))

            plt.legend(self.case_path_list)
            plt.xlabel('Time (ns)')
            plt.ylabel("%s Count" % specie)
            plt.savefig(os.path.join(save_path, '%s.png' % specie))
            plt.close()

    def compare_rxn_path(self):
        save_path = os.path.join(self.work_folder, 'compare_rxn')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        time_list = self.case_list[0].step_list
        rxn_frame_list = list(
            map(lambda x: RxnPath(data=x.rxn_list, index=time_list,
                                  save_path=os.path.join(self.work_folder, x.case_folder)), self.case_list))

        # compare path
        rxn_path_frame = pd.DataFrame([pd.Series(x.summed_path_dict) for x in rxn_frame_list],
                                      index=self.case_path_list).fillna(0).T.astype(int)
        rxn_path_frame['sum'] = rxn_path_frame.sum(axis=1)
        rxn_path_frame.sort_values(by='sum', ascending=False, inplace=True)
        rxn_path_frame.to_csv(os.path.join(save_path, 'sum_path.csv'))


class LmpCaseAverage:
    """
    the lmp results need average to reduce system uncertainty.
    sum the case results and average it.
    """

    def __init__(self, work_folder: str):
        self.work_folder = work_folder
        self.save_folder = os.path.join(self.work_folder, 'average')
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
        self.average_rate()
        self.trim_case_len_()
        self.specie_trace = None

    @property
    @lru_cache(maxsize=None)
    def case_path_list(self) -> List[str]:
        folder_list = [folder for folder in list(os.listdir(self.work_folder))
                       if os.path.exists(os.path.join(self.work_folder, folder, 'lmp_case.pickle'))]
        folder_list.sort()
        return folder_list

    @property
    @lru_cache(maxsize=None)
    def case_list(self) -> List[LmpCase]:
        """list of LmpCase object"""
        case_list = list(
            map(lambda x: pickle.load(Path(os.path.join(self.work_folder, x, 'lmp_case.pickle')).open('rb'))
                , self.case_path_list))
        return case_list

    def trim_case_len_(self):
        """align the case length"""
        minima_len = min([len(item.step_list) for item in self.case_list])
        for case in self.case_list:
            del case.step_list[minima_len:]
            del case.bond_list[minima_len:]
            del case.specie_list[minima_len:]
            del case.rxn_list[minima_len:]
        return minima_len

    def plot_bond(self):

        data = None
        for item in self.case_list:
            if data is None:
                data = item.bond_list
            else:
                data = list(map(add, data, item.bond_list))

        time_index = np.array(self.case_list[0].step_list)
        bond_trace = RxnBondTrace(data=data, index=time_index, save_path=self.save_folder)
        bond_trace.frame //= len(self.case_list)
        bond_trace.plot()

    def plot_specie(self):
        data = None
        for item in self.case_list:
            if data is None:
                data = item.specie_list
            else:
                data = list(map(add, data, item.specie_list))
        time_index = np.array(self.case_list[0].step_list)
        specie_trace = RxnSpecieTrace(data=data, index=time_index, save_path=self.save_folder)
        specie_trace.frame //= len(self.case_list)
        self.specie_trace = specie_trace
        specie_trace.plot()

    @staticmethod
    def time_step_index(step_list, interval=0.1) -> list:
        """get the index of each interval unit: ns"""
        interval = interval * 1e6
        step_array = np.array(step_list)
        step_interval = int(np.ceil(interval / (step_array[1] - step_array[0])))
        if step_interval > len(step_array):
            return []
        key_index_list = list(range(len(step_array)))
        return key_index_list[::step_interval]

    def plot_path(self):

        data = None
        for item in self.case_list:
            if data is None:
                data = item.rxn_list
            else:

                # data = list(map(max_counter, data, item.rxn_list))
                data = list(map(add, data, item.rxn_list))

        step_list = np.array(self.case_list[0].step_list)
        rxn_list = data

        if METAL[0]:
            if METAL_NAME[0] in RXN_START_SPECIE[0]:
                start_specie=RXN_START_SPECIE[0]
            else:
                start_specie=METAL_NAME[0]
                
            interval = 0.1
            rxn_save_path = os.path.join(self.save_folder, "rxn_metal")
            key_index_list = self.time_step_index(step_list, interval=interval)

            if not os.path.exists(rxn_save_path):
                os.mkdir(rxn_save_path)

            rxn_path = RxnPath(data=rxn_list, index=step_list, save_path=rxn_save_path)

            for threshold in PATH_THRESHOLD:
                rxn_path.plot_path(start_specie=start_specie, threshold=threshold)

            rxn_save_path = os.path.join(self.save_folder, "rxn_metal", 'interval')
            if not os.path.exists(rxn_save_path):
                os.mkdir(rxn_save_path)
            rxn_time_list = []
            sum_rxn_time_list = []
            rxn_time_index = []

            for index_a, index_b in zip(key_index_list[0:-1], key_index_list[1:]):
                save_name = '%.1f-%.1f' % (step_list[index_a] / 1e6, step_list[index_b] / 1e6)
                rxn_time_index.append(save_name)
                rxn_save_path = os.path.join(self.save_folder, "rxn_metal", 'interval', save_name)
                if not os.path.exists(rxn_save_path):
                    os.mkdir(rxn_save_path)
                rxn_path = RxnPath(data=rxn_list[index_a:index_b], index=step_list[index_a:index_b],
                                   save_path=rxn_save_path)
                rxn_time_list.append(rxn_path.path_dict)
                sum_rxn_time_list.append(rxn_path.summed_path_dict)

                for threshold in PATH_THRESHOLD:
                    rxn_path.plot_path(start_specie=start_specie, threshold=threshold)

            rxn_interval_df = pd.DataFrame(data=rxn_time_list, index=rxn_time_index).fillna(0).T.to_csv(
                os.path.join(self.save_folder, "rxn_metal", 'rxn_interval.csv'))
            sum_interval_df = pd.DataFrame(data=sum_rxn_time_list, index=rxn_time_index).fillna(0).T.to_csv(
                os.path.join(self.save_folder, "rxn_metal", 'sum_interval.csv'))
            return None

        # average the data
        # sample = len(self.case_list)
        # for item in data:
        #     for key, val in item.items():
        #         item[key] = int(val / sample)

        if not self.specie_trace.frame.get(RXN_START_SPECIE, None):
            return None
        start_specie_trace = self.specie_trace.frame[RXN_START_SPECIE].to_numpy()
        consume_list = [(-1, -1)]
        for consume_ratio in 0.25, 0.5, 0.75, 0.90, 0.95, 0.99:
            consume_index = np.where(start_specie_trace < start_specie_trace[0] * (1 - consume_ratio))[0]
            if len(consume_index) > 1:
                consume_list.append((consume_ratio, consume_index[0]))

        for consume_ratio, consume_index in consume_list:
            rxn_save_path = os.path.join(self.save_folder, "rxn_consume_%d" % (consume_ratio * 100))
            if not os.path.exists(rxn_save_path):
                os.mkdir(rxn_save_path)
            rxn_path = RxnPath(data=rxn_list[:consume_index], index=step_list[:consume_index],
                               average_val=1, save_path=rxn_save_path)
            for specie in RXN_START_SPECIE:
                for threshold in PATH_THRESHOLD:
                    threshold *= max(len(self.case_list) // 4, 1)
                    rxn_path.plot_path(start_specie=specie, threshold=threshold)

            for specie in RXN_START_SPECIE + FLUX_SPECIE:
                for threshold in PATH_THRESHOLD:
                    threshold *= max(len(self.case_list) // 4, 1)
                    rxn_path.plot_flux(specie=specie, threshold=threshold)

    def plot_all(self):
        self.plot_bond()
        self.plot_specie()
        self.plot_path()

    def average_rate(self):
        if METAL[0]:
            return None
        pd_list = [pd.read_csv(os.path.join(self.work_folder, item, 'rate.csv'), index_col=0) for item in
                   self.case_path_list if
                   os.path.exists(os.path.join(self.work_folder, item, 'rate.csv'))]
        common_reaction = set(pd_list[0].index.to_list())
        for i in range(1, len(pd_list)):
            common_reaction = common_reaction.intersection(set(pd_list[i].index.to_list()))
        common_reaction = list(common_reaction)
        rate_frame = pd_list[0].loc[common_reaction]
        for i in range(1, len(pd_list)):
            rate_frame += pd_list[i].loc[common_reaction]
        rate_frame /= len(pd_list)
        rate_frame.to_csv(os.path.join(self.save_folder, 'rate.csv'), float_format='%.2e')


def max_counter(counter_A: Counter, counter_B: Counter):
    """This method is failed, already departure
    To avoid average omit key reactions, instead average of reaction path,
     MAX() is used."""
    maxCounter = counter_A
    for key, item in counter_B.items():
        if key not in maxCounter:
            maxCounter[key] = item
        else:
            maxCounter[key] = max(maxCounter[key], item)
    return maxCounter


if __name__ == '__main__':
    # single case test
    import pprint

    LmpCase(case_folder='/home/hao/code/reax_reaction/example/5A2')

    with open('/home/hao/code/reax_reaction/example/aluminum/lmp_case.pickle', 'rb') as f:
        lmp_case = pickle.load(f)
    lmp_case.plot_all()

    # LmpCaseAverage(work_folder='example/average').plot_path()
