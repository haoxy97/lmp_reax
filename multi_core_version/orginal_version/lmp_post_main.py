#!/usr/bin/env python3

"""
main program of lmp post
"""
import os
from glob import glob
import time
from lmpframe import BondFrame, init_BondFrame, lz4_2_block
from lmpdatatrace import FrameTrace, FrameTraceInitializer
from lmpcase import LmpCase, LmpCaseAverage
import utility
from multiprocessing import Pool
import argparse
import lmp_env

block_size=lmp_env.BLOCK_SIZE[0]

def parse_block_iter(block_iter: list):
    """function use for map parallel"""
    return list(map(BondFrame, block_iter))


def parse_batch(lz4_file: str, use_cache=True, block_size=block_size, worker=1):
    t0 = time.time()
    save_name = os.path.join(os.path.dirname(lz4_file).rstrip('bond'), 'trace',
                             os.path.splitext(os.path.basename(lz4_file))[0] + '.pickle')
    if os.path.exists(save_name) and use_cache:
        print('Parsed %s, load cache' % lz4_file)
        return 1
    batch_stat_trace = None

    # use block to split lz4 batch to reduce mem usage
    for block in lz4_2_block(lz4_file, block_size=block_size):
        block_iter_list = []
        worker_frame_num = min(1000 // worker, 20 * worker)
        for index in range(0, len(block), worker_frame_num):
            block_iter_list.append(block[index:index + worker_frame_num])

        with Pool(processes=worker) as p:
            frame_block = sum(list(p.imap(parse_block_iter, block_iter_list)), [])

        if batch_stat_trace is None:
            batch_stat_trace = FrameTraceInitializer(batch=frame_block, sample_avg=block_size // 5).to_FrameTrace()
        else:
            batch_stat_trace += FrameTraceInitializer(batch=frame_block, sample_avg=block_size // 5).to_FrameTrace()
        del frame_block[:]  # free memo release
        del block_iter_list[:]  # free memo release

    batch_stat_trace.to_pickle(save_name=save_name)
    print('Parsed %s,%6.2f' % (lz4_file, time.time() - t0))
    return 0


def parse_case(job_path: str, use_cache=True, worker=8):
    """parse lmp bond lz4 file to organized pickle object for LmpCase post job"""
    data_file = glob(os.path.join(job_path, "*.data"))
    if len(data_file) < 1:
        print('Error: can not find data structure file')
        exit(1)
    data_file = data_file[0]
    trace_folder = os.path.join(job_path, "trace")
    if not os.path.exists(trace_folder):
        os.mkdir(trace_folder)
    lz4_files = sorted(glob(os.path.join(job_path, "bond/*.lz4")))
    # init bondFrame
    init_BondFrame(lmp_data_path=data_file)
    # loop
    for lz4_file in lz4_files:
        parse_batch(lz4_file, use_cache=use_cache, worker=worker)


def argv_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-s', '--specie', help='specie as rxn path start, [C10H8__0800r6-2]', type=str,
                        default='C10H8__0800r6-2')

    parser.add_argument('-t', '--threshold', help='threshold for add species plot, [0]', type=int,
                        default=0)
    parser.add_argument('-c', '--core', help='parallel cores, [8]', type=int, default=8)
    parser.add_argument('-b', '--block', help='block size, dpi of cases, [1000]', type=int, default=1000)
    parser.add_argument('-p', '--path', help='job path, [./]', default='./', type=str)
    parser.add_argument('-n', '--NoLoadCache', help='No use cache, [False]', default=False, action='store_true')
    parser.add_argument('-f', '--figurePlot', help='Plot figure, [True]', default=True, action='store_false')
    parser.add_argument('-d', '--debug', help='debug mode, [False]', default=False, action='store_true')
    parser.add_argument('-a', '--averageCase', help='average multi cases, [False]', default=False, action='store_true')
    parser.add_argument('-m', '--metal', help='calculate specie in metal mode, [False]', default=False,
                        action='store_true')
    parser.add_argument('-mn', '--metalname', help='set metal name, [Al]', type=str,
                        default='Al')
    parser.add_argument('-mi', '--metalID', help='set metal id, [1]', type=int,
                        default=1)

    parser.add_argument('-sf', '--skipFrame', help='skip Frame, [1]', type=int,
                        default=1)
    parser.print_help()
    job_argv = parser.parse_args()
    return job_argv


if __name__ == '__main__':
    utility.set_graph_format()
    argv = argv_parser()
    lmp_env.RXN_START_SPECIE[0] = argv.specie
    lmp_env.SKIP_FRAME[0] = argv.skipFrame
    lmp_env.BLOCK_SIZE[0] = argv.block
    cache_flag = not argv.NoLoadCache

    print('skip frame %d' % argv.skipFrame)

    if argv.metal:
        lmp_env.METAL[0] = True
        lmp_env.METAL_NAME[0] = argv.metalname
        lmp_env.METAL_ID[0] = argv.metalID
        lmp_env.type2element_str_dict[argv.metalID] = argv.metalname
        print("running in metal mode, metal id %d, metal name %s" % (lmp_env.METAL_ID[0], lmp_env.METAL_NAME[0]))

    if argv.threshold > 0:
        lmp_env.PATH_THRESHOLD.append(argv.threshold)

    # debug section
    if argv.debug:
        print('debug mode')
        import sys

        argv.path = '/home/hao/run/MD/example/combustion'
        parse_case(argv.path, use_cache=True, worker=argv.core)
        case = LmpCase(argv.path)
        case.plot_all()
        sys.exit(1)

    if not argv.averageCase:

        if os.path.exists(os.path.join(argv.path, 'log.lammps')):
            parse_case(argv.path, use_cache=cache_flag, worker=argv.core)
            if argv.figurePlot:
                LmpCase(argv.path).plot_all()
            else:
                LmpCase(argv.path)
        else:
            print('current folder is not a lammps case folder')

    else:
        print('Average multi cases')
        case_folder_list = [os.path.join(argv.path, folder) for folder in list(os.listdir(argv.path))
                            if os.path.exists(os.path.join(argv.path, folder, 'log.lammps'))]
        list(map(LmpCase, [x for x in case_folder_list]))
        print('Statistics the results')
        lmp_average = LmpCaseAverage(work_folder=argv.path)
        lmp_average.plot_all()
