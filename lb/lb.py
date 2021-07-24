import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from .utils import profile

C = np.ascontiguousarray([[0, 0, 1, 0, -1, 1, 1, -1, -1],  # y
                          [0, 1, 0, -1, 0, 1, -1, -1, 1]]).T  # x
C.setflags(write=False)

W = np.ascontiguousarray([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T
W.setflags(write=False)

OPPOSITE_IDXS = np.ascontiguousarray([0,3,4,1,2,7,8,5,6])
OPPOSITE_IDXS.setflags(write=False)


def calculate_density(f):
    density = np.sum(f, axis=-1)
    return density


def calculate_velocity_field(f, density):
    velocity_field = np.dot(f, C) / (density[:, :, None] + np.finfo(float).eps)
    return velocity_field


def calculate_equilibrium_distribution(density, velocity_field):
    c_dot_vf = (velocity_field[:, :, :, None] * C.T[None, None])
    c_dot_vf = np.sum(c_dot_vf, axis=2)
    vf_norm_square = np.sum(velocity_field**2, axis=2)[:, :, None]
    feq = W * (density[:, :, None] * (1 + 3 * c_dot_vf + 4.5*c_dot_vf**2 - 1.5*vf_norm_square))
    return feq


def split(array, n, axis, rank):
    '''Split with padding.'''
    arrays = np.array_split(array, n, axis=axis)
    array = np.concatenate([np.take(arrays[rank-1], [-1], axis=axis),
        arrays[rank],
        np.take(arrays[(rank+1) % n], [0], axis=axis)], axis=axis)
    return array


class LatticeBoltzmann():
    def __init__(self, density, velocity_field):
        comm = MPI.COMM_WORLD
        self.n_workers = comm.Get_size()
        self.rank = comm.Get_rank()
        self.h, self.w = density.shape

        self._density = split(density, self.n_workers, 0, self.rank)
        self._velocity_field = split(velocity_field, self.n_workers, 0, self.rank)
        self._f = calculate_equilibrium_distribution(self._density, self._velocity_field)

        self.f = np.zeros((self.h, self.w, 9), dtype=np.float32)
        self.velocity_field = np.zeros((self.h, self.w, 2), dtype=np.float32)
        self.density = np.zeros((self.h, self.w), dtype=np.float32)

        self.gather_f()

    def stream(self):
        for i in range(9):
            self._f[:, :, i] = np.roll(self._f[:, :, i], C[i], axis=(0, 1))
        return

    def collide(self, tau=0.6):
        self._density = calculate_density(self._f)
        self._velocity_field = calculate_velocity_field(self._f, self._density)
        feq = calculate_equilibrium_distribution(self._density, self._velocity_field)
        self._f += 1/tau * (feq - self._f)

    def stream_and_collide(self, tau=0.6):
        self.partial_update_f()
        self.stream()
        self.collide(tau)
        self.gather_f()

    def _gather(self, name):
        array = np.ascontiguousarray(getattr(self, f'_{name}')[1:-1], dtype=np.float32)
        MPI.COMM_WORLD.Allgatherv([array, MPI.FLOAT], [getattr(self, f'{name}'), MPI.FLOAT])

    def partial_update_f(self):
        self._f = split(self.f, self.n_workers, 0, self.rank)

    def gather_f(self):
        self._gather('f')

    def gather_density(self):
        self._gather('density')

    def gather_velocity_field(self):
        self._gather('velocity_field')

    def plot(self, ax=None):
        if ax is not None:
            _plt = ax
        else:
            _plt = plt
        v = np.sqrt(self.velocity_field[:, :, 0]**2 + self.velocity_field[:, :, 1]**2)
        v = np.ma.masked_where((v < 0.0), v)
        _plt.imshow(v, cmap = 'RdBu_r', vmin=0, interpolation = 'spline16')

        # vorticity = (np.roll(self.velocity_field[1], -1, axis=0) - np.roll(self.velocity_field[1], 1, axis=0)) - (np.roll(self.velocity_field[0], -1, axis=1) - np.roll(self.velocity_field[0], 1, axis=1))
        # # vorticity[self.velocity_field.sum(axis=0) == 0] = np.nan
        # cmap = plt.cm.get_cmap("bwr").copy()
        # cmap.set_bad('black')
        # plt.imshow(vorticity, cmap='bwr')
