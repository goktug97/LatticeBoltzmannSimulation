from abc import ABC, abstractmethod

import numpy as np

import lb


class Boundary(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass


class WallBoundary(Boundary):
    def __init__(self, idxs):
        self.idxs = idxs

    def forward(self, f):
        self.cache = f[self.idxs].copy()

    def backward(self, f):
        f[self.idxs] = self.cache[:, lb.OPPOSITE_IDXS]


class MovingWallBoundary(Boundary):
    def __init__(self, idxs, wall_velocity):
        self.wall_velocity = wall_velocity
        self.idxs = idxs

    def forward(self, f):
        self.cache = f[self.idxs].copy()

    def backward(self, f):
        density = lb.calculate_density(f[self.idxs])
        momentum = 6 * (lb.C @ self.wall_velocity) * (lb.W * density[:, None])
        f[self.idxs] = self.cache[:, lb.OPPOSITE_IDXS] + momentum


class HorizontalInletOutletBoundary(Boundary):
    def __init__(self, ny, velocity):
        self.inlet_f = lb.calculate_equilibrium_distribution(
                np.ones((ny, 1)), velocity).squeeze()

    def forward(self, f):
        self.cache = f[:, -2, [3, 6, 7]].copy()

    def backward(self, f):
        f[:, -1, [3, 6, 7]] = self.cache
        f[:, 0] = self.inlet_f
