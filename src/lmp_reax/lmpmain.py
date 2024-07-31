#!/usr/bin/env python3

"""
main program of lmp post
"""
import os
from glob import glob
from .lmpframe import BondFrame, init_BondFrame, gz_2_block
from .lmpdatatrace import FrameTraceInitializer
from .lmpcase import LmpCase
from multiprocessing import Pool
from . import lmp_env
from tqdm import tqdm

block_size = lmp_env.BLOCK_SIZE[0]


def parse_batch(zip_file: str, use_cache=True, block_size=block_size):
    save_name = os.path.join(os.path.dirname(zip_file).rstrip('bond'), 'trace',
                             os.path.splitext(os.path.basename(zip_file))[0] + '.pickle')
    if os.path.exists(save_name) and use_cache:
        print('Parsed %s, load cache' % zip_file)
        return 0

    batch_stat_trace = None
    # use block to split lz4 batch to reduce mem usage
    blocks=gz_2_block(zip_file, block_size=block_size)
    for block in tqdm(blocks, desc='Parsing %s' % zip_file,total=len(blocks)):
        frame_block = list(map(BondFrame, block))
        if batch_stat_trace is None:
            batch_stat_trace = FrameTraceInitializer(batch=frame_block, sample_avg=block_size // 5).to_FrameTrace()
        else:
            batch_stat_trace += FrameTraceInitializer(batch=frame_block, sample_avg=block_size // 5).to_FrameTrace()
        del frame_block[:]  # free memo release
    batch_stat_trace.to_pickle(save_name=save_name)
    return 0


def parse_case(job_path: str, use_cache=True, block_size=block_size):
    data_file = glob(os.path.join(job_path, "*.data"))
    if len(data_file) < 1:
        print('Error: can not find data structure file')
        exit(1)
    data_file = data_file[0]
    trace_folder = os.path.join(job_path, "trace")
    if not os.path.exists(trace_folder):
        os.mkdir(trace_folder)
    zip_files = sorted(glob(os.path.join(job_path, "bond/*.gz")))
    init_BondFrame(lmp_data_path=data_file)
    for zip_file in zip_files:
        parse_batch(zip_file, use_cache=use_cache, block_size=block_size)


def run(case_path = './'):
    parse_case(case_path, use_cache=True, block_size=lmp_env.BLOCK_SIZE[0])
    case = LmpCase(case_path)
    case.plot_all()