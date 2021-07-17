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
    density = np.sum(f, axis=2)
    return density


def calculate_velocity_field(f, density):
    velocity_field = np.dot(f, C) / (density[:, :, None] + np.finfo(float).eps)
    return velocity_field.transpose(2, 0, 1)


def calculate_equilibrium_distribution(density, velocity_field):
    c_dot_vf = (velocity_field[:, :, :, None] * C.T[:, None, None, :])
    c_dot_vf = np.sum(c_dot_vf, axis=0)
    vf_norm_square = np.sum(velocity_field**2, axis=0)[:, :, None]
    (1 + 3 * c_dot_vf + 4.5*c_dot_vf**2 - 1.5*vf_norm_square)
    feq = W * (density[:, :, None] * (1 + 3 * c_dot_vf + 4.5*c_dot_vf**2 - 1.5*vf_norm_square))
    return feq


class LatticeBoltzmann():
    def __init__(self, density, velocity_field):
        self.f = calculate_equilibrium_distribution(density, velocity_field)
        self.density = density
        self.velocity_field = velocity_field
        comm = MPI.COMM_WORLD
        self.n_workers = comm.Get_size()
        self.rank = comm.Get_rank()

    def stream(self):
        for i in range(9):
            self.f[:, :, i] = np.roll(self.f[:, :, i], C[i], axis=(0, 1))
        return

        f = np.transpose(self.f, (1, 2, 0))
        lattices = np.split(f, self.n_workers, axis=0)
        lattice = np.concatenate([lattices[self.rank-1][-1:],
                             lattices[self.rank],
                             lattices[(self.rank+1) % self.n_workers][:1]], axis=0)
        for i in range(9):
            lattice[:, :, i] = np.roll(lattice[:, :, i], C[i], axis=(0, 1))
        lattice = np.ascontiguousarray(lattice[1:-1], dtype=np.float32)
        f = np.ascontiguousarray(f, dtype=np.float32)
        MPI.COMM_WORLD.Allgatherv([lattice, MPI.FLOAT], [f, MPI.FLOAT])
        self.f = np.transpose(f, (2, 0, 1))

    def collide(self, tau=0.6):
        self.density = calculate_density(self.f)
        self.velocity_field = calculate_velocity_field(self.f, self.density)
        feq = calculate_equilibrium_distribution(self.density, self.velocity_field)
        self.f += 1/tau * (feq - self.f)

    def plot(self):
        v = np.sqrt(self.velocity_field[0]**2 + self.velocity_field[1]**2)
        v = np.ma.masked_where((v < 0.0), v)
        plt.imshow(v, cmap = 'RdBu_r', vmin=0, interpolation = 'spline16')

        # vorticity = (np.roll(self.velocity_field[1], -1, axis=0) - np.roll(self.velocity_field[1], 1, axis=0)) - (np.roll(self.velocity_field[0], -1, axis=1) - np.roll(self.velocity_field[0], 1, axis=1))
        # # vorticity[self.velocity_field.sum(axis=0) == 0] = np.nan
        # cmap = plt.cm.get_cmap("bwr").copy()
        # cmap.set_bad('black')
        # plt.imshow(vorticity, cmap='bwr')
