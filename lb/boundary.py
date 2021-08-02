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


class MovingTopWallBoundary(Boundary):
    def __init__(self, wall_velocity, cs=1/np.sqrt(3)):
        self.wall_velocity = wall_velocity
        self.cs = cs

    def forward(self, f):
        self.cache = f[0].copy()

    def backward(self, f):
        density = lb.calculate_density(f[0])
        multiplier = 2 * (1/self.cs) ** 2
        momentum = multiplier * (lb.C @ self.wall_velocity) * (lb.W * density[:, None])

        f[0, :, 2] = self.cache[:, 4]
        f[0, :, 5] = self.cache[:, 7] - momentum[:, 7]
        f[0, :, 6] = self.cache[:, 8] - momentum[:, 8]


class HorizontalInletOutletBoundary(Boundary):
    def __init__(self, ny, velocity):
        self.inlet_f = lb.calculate_equilibrium_distribution(
                np.ones((ny, 1), dtype=np.float32), velocity).squeeze()

    def forward(self, f):
        self.cache = f[:, -2, [3, 6, 7]].copy()

    def backward(self, f):
        f[:, -1, [3, 6, 7]] = self.cache
        f[:, 0] = self.inlet_f
