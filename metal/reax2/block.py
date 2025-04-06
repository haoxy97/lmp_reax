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
    # pickle.dump(FrameEnv(n_atom, type_array, element_array, mat), Path(cache).open('wb'))
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
            # load from cache
            frame_env = init_frame_env(frame_env)
        steps = []
        speice_list = []
        for step, box, cord in self.nc_parser(nc_file):
            frame = Frame(step, cord, box, frame_env)
            steps.append(frame.step)
            speice_list.append(frame.specie_id_list)
        block = Block(np.array(steps), np.array(speice_list))
        block.save(file_path=nc_file+'.npz')
        # print('Parsed: %s' % nc_file)
        return block

    def parse_packed(self, args):
        nc_file, frame_env, use_cache = args
        cache = nc_file+'.npz'
        if Path(cache).exists() and use_cache:
            return Block.load(cache)
        if isinstance(frame_env, str):
            # load from cache
            frame_env = init_frame_env(frame_env)
        steps = []
        speice_list = []
        for step, box, cord in self.nc_parser(nc_file):
            frame = Frame(step, cord, box, frame_env)
            steps.append(frame.step)
            speice_list.append(frame.specie_id_list)
        block = Block(np.array(steps), np.array(speice_list))
        block.save(file_path=nc_file+'.npz')
        # print('Parsed: %s' % nc_file)
        return block


if __name__ == "__main__":
    nc_file = '../example/1/1.nc'
    # env = init_frame_env(nc_file)
    env = None
    b = NcParser().parse(nc_file, frame_env=env, use_cache=False)
    (b+b).save_csv('specie_list.csv')
