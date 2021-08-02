import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

C = np.ascontiguousarray(
        np.array([[0, 0, 1, 0, -1, 1, 1, -1, -1],
                  [0, 1, 0, -1, 0, 1, -1, -1, 1]]).T)
C.setflags(write=False)

W = np.ascontiguousarray(
        np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T,
        dtype=np.float32)
W.setflags(write=False)

OPPOSITE_IDXS = np.ascontiguousarray([0, 3, 4, 1, 2, 7, 8, 5, 6])
OPPOSITE_IDXS.setflags(write=False)


def calculate_density(f):
    density = np.sum(f, axis=-1)
    return density


def calculate_velocity_field(f, density):
    velocity_field = np.dot(f, C) / (density[:, :, None] + np.finfo(np.float32).eps)
    return velocity_field


def calculate_equilibrium_distribution(density, velocity_field):
    c_dot_vf = (velocity_field[:, :, :, None] * C.T[None, None])
    c_dot_vf = np.sum(c_dot_vf, axis=2)
    vf_norm_square = np.sum(velocity_field**2, axis=2)[:, :, None]
    feq = W * (density[:, :, None] * (1 + 3 * c_dot_vf + 4.5*c_dot_vf**2 - 1.5*vf_norm_square))
    return feq


class LatticeBoltzmann():
    def __init__(self, density, velocity_field):
        comm = MPI.COMM_WORLD
        self.n_workers = comm.Get_size()
        self.rank = comm.Get_rank()
        self._parallel_vars = []

        assert len(density.shape) == 2
        assert len(velocity_field.shape) == 3
        assert density.shape == velocity_field.shape[:2]
        assert velocity_field.shape[2] == 2

        self.h, self.w = density.shape

        density, velocity_field = density.astype(np.float32), velocity_field.astype(np.float32)

        f = calculate_equilibrium_distribution(density, velocity_field)

        self._parallel_var('velocity_field', velocity_field)
        self._parallel_var('density', density)
        self._parallel_var('f', f)

        self.boundaries = []

    def stream(self):
        for i in range(9):
            self._s_f[:, :, i] = np.roll(self._s_f[:, :, i], C[i], axis=(0, 1))
        return

    def collide(self, tau=0.6):
        self._s_density = calculate_density(self._s_f)
        self._s_velocity_field = calculate_velocity_field(self._s_f, self._s_density)
        feq = calculate_equilibrium_distribution(self._s_density, self._s_velocity_field)
        self._s_f += 1/tau * (feq - self._s_f)

    def stream_and_collide(self, tau=0.6):
        for boundary in self.boundaries:
            boundary.forward(self.f)
        self._partial_update_f()
        self.stream()
        self.collide(tau)
        for parallel_var in self._parallel_vars:
            setattr(self, f'_{parallel_var}_calculated', False)
        for boundary in self.boundaries:
            boundary.backward(self.f)

    def add_boundary(self, boundary):
        self.boundaries.append(boundary)

    def _gather(self, name):
        array = getattr(self, f'_s_{name}')[1:-1]
        array = np.ascontiguousarray(array, dtype=np.float32)
        x = getattr(self, f'_{name}')
        cheight, width, c = x.shape
        if cheight % self.n_workers:
            height = cheight + cheight % self.n_workers
            x = np.ascontiguousarray(np.zeros((height, width, c)), dtype=np.float32)
            MPI.COMM_WORLD.Allgatherv([array, MPI.FLOAT], [x, MPI.FLOAT])
            array = getattr(self, f'_{name}')
            array[:] = x[:cheight]
        else:
            MPI.COMM_WORLD.Allgatherv([array, MPI.FLOAT], [x, MPI.FLOAT])

        setattr(self, f'_{name}_calculated', True)
        array = getattr(self, f'_{name}')

    def _split(self, array):
        arrays = np.array_split(array, self.n_workers, axis=0)
        array = np.concatenate([arrays[self.rank-1][-1:],
                                arrays[self.rank],
                                arrays[(self.rank+1) % self.n_workers][:1]])
        return array

    def _parallel_var(self, name, value):
        setattr(self, f'_s_{name}', self._split(value))
        setattr(self, f'_{name}', np.zeros_like(value, dtype=np.float32))
        setattr(self, f'_{name}_calculated', False)
        setattr(LatticeBoltzmann, name, self._parallel_property(name))
        setattr(LatticeBoltzmann, f'gather_{name}', lambda self: self._gather(name))
        self._parallel_vars.append(name)

    @staticmethod
    def _parallel_property(name):
        def func(self):
            if not getattr(self, f'_{name}_calculated'):
                self._gather(name)
            return getattr(self, f'_{name}')
        func.__name__ = name
        return property(func)

    def _partial_update_f(self):
        self._s_f = self._split(self.f)

    def plot(self, ax=None):
        if ax is not None:
            _plt = ax
        else:
            _plt = plt
        v = np.sqrt(self.velocity_field[1:-1, 1:-1, 0]**2 +
                self.velocity_field[1:-1, 1:-1, 1]**2)
        v = np.ma.masked_where((v < 0.0), v)
        _plt.imshow(v, cmap='RdBu_r', vmin=0, interpolation='spline16')

    def streamplot(self, ax=None):
        if ax is not None:
            _plt = ax
        else:
            _plt = plt
        x, y = np.meshgrid(np.arange(self.w-2), np.arange(self.h-2))
        _plt.streamplot(x, y, self.velocity_field[1:-1, 1:-1, 1],
                self.velocity_field[1:-1, 1:-1, 0])
