import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from .utils import *
from .boundary import HorizontalInletOutletBoundary


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

        # Create parallel variables
        self._parallel_var('velocity_field', velocity_field)
        self._parallel_var('density', density)
        self._parallel_var('f', f)
        self._parallel_var('feq', f)

        self.boundaries = []

    def stream(self):
        for i in range(9):
            self._s_f[:, :, i] = np.roll(self._s_f[:, :, i], C[i], axis=(0, 1))

    def collide(self, tau=0.6):
        self._s_density = calculate_density(self._s_f)
        self._s_velocity_field = calculate_velocity_field(self._s_f, self._s_density)
        self._s_feq = calculate_equilibrium_distribution(self._s_density, self._s_velocity_field)
        self._s_f += 1/tau * (self._s_feq - self._s_f)

    def stream_and_collide(self, tau=0.6):
        """A full LBM step with boundary handling."""
        for boundary in self.boundaries:
            if isinstance(boundary, HorizontalInletOutletBoundary):
                boundary.forward(self.f, self.feq, self.velocity_field)
            else:
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
        """Gather split arrays for `name` to shared array if it is not updated.
        See: `_parallel_var` function for definitions."""
        if not getattr(self, f'_{name}_calculated'):
            array = getattr(self, f'_s_{name}')[1:-1]
            array = np.ascontiguousarray(array, dtype=np.float32)
            x = getattr(self, f'_{name}')
            # Handle edge where height is not divisible by number of workers.
            if x.shape[0] % self.n_workers:
                height = int(np.ceil(x.shape[0] / self.n_workers) * self.n_workers)
                x = np.ascontiguousarray(np.zeros((height, *x.shape[1:])), dtype=np.float32)
                MPI.COMM_WORLD.Allgatherv([array, MPI.FLOAT], [x, MPI.FLOAT])
                array = getattr(self, f'_{name}')
                array[:] = x[:array.shape[0]]
            else:
                MPI.COMM_WORLD.Allgatherv([array, MPI.FLOAT], [x, MPI.FLOAT])

            setattr(self, f'_{name}_calculated', True)
            array = getattr(self, f'_{name}')

    def _split(self, array):
        """Split given array for MPI processes
        but keeping the last row and first row of the previous and the next processes'
        split array respectively for streaming operation."""
        arrays = np.array_split(array, self.n_workers, axis=0)
        array = np.concatenate([arrays[self.rank-1][-1:],
                                arrays[self.rank],
                                arrays[(self.rank+1) % self.n_workers][:1]])
        return array

    def _parallel_var(self, name, value):
        """Create a parallel variable for `name` with the given initial `value`.
        Creates:
            _name: Shared value for name.
            name: Property, when it is accessed, it updates the shared value and returns it.
            _s_name: Split value for name. Every MPI processes hold a different part of the shared value.
            _name_calculated: Indicates whether the current shared value is updated or not.
            gather_name: Function that updates the shared value.
            """
        setattr(self, f'_s_{name}', self._split(value))
        setattr(self, f'_{name}', np.zeros_like(value, dtype=np.float32))
        setattr(self, f'_{name}_calculated', False)
        setattr(LatticeBoltzmann, name, self._parallel_property(name))
        setattr(LatticeBoltzmann, f'gather_{name}', lambda self: self._gather(name))
        self._parallel_vars.append(name)

    @staticmethod
    def _parallel_property(name):
        """Create a property for `name` that gathers the value
        from every MPI processes if not calculated."""
        def func(self):
            if not getattr(self, f'_{name}_calculated'):
                self._gather(name)
            return getattr(self, f'_{name}')
        func.__name__ = name
        return property(func)

    def _partial_update_f(self):
        """Update split value of self._f"""
        self._s_f = self._split(self.f)

    def plot(self, ax=None):
        """Plot velocity field."""
        if ax is not None:
            _plt = ax
        else:
            _plt = plt
        for boundary in self.boundaries:
            boundary.update_velocity(self.velocity_field)
        v = np.sqrt(self.velocity_field[:, :, 0]**2 +
                    self.velocity_field[:, :, 1]**2)
        _plt.imshow(v, cmap='RdBu_r', vmin=0, interpolation='spline16')

    def streamplot(self, ax=None):
        """Plot streamplot of the velocity field."""
        if ax is not None:
            _plt = ax
        else:
            _plt = plt
        for boundary in self.boundaries:
            boundary.update_velocity(self.velocity_field)
        x, y = np.meshgrid(np.arange(self.w), np.arange(self.h))
        _plt.streamplot(x, y, self.velocity_field[:, :, 1], self.velocity_field[:, :, 0])
