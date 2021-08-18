from abc import ABC, abstractmethod

import numpy as np

import lb


class Boundary(ABC):
    @abstractmethod
    def forward(self):
        """Called before streaming operation to cache the pre-stream values."""
        pass

    @abstractmethod
    def backward(self):
        """Called after the collision operation to apply boundary conditions."""
        pass

    def update_velocity(self, velocity_field):
        pass


class RigidObject(Boundary):
    def __init__(self, mask):
        self.mask = mask

    def forward(self, f):
        self.cache = f[self.mask]

    def backward(self, f):
        f[self.mask] = self.cache[:, lb.OPPOSITE_IDXS]

    def update_velocity(self, velocity_field):
        velocity_field[self.mask] = 0.0


class WallBoundary(Boundary):
    def __init__(self):
        self.idxs = None


class RightWallBoundary(WallBoundary):
    def __init__(self):
        self.idxs = [3, 6, 7]

    def forward(self, f):
        self.cache = f[:, -2]

    def backward(self, f):
        f[:, -1, self.idxs] = self.cache[:, lb.OPPOSITE_IDXS[self.idxs]]

    def update_velocity(self, velocity_field):
        velocity_field[:, -1] = 0.0


class LeftWallBoundary(WallBoundary):
    def __init__(self):
        self.idxs = [1, 5, 8]

    def forward(self, f):
        self.cache = f[:, 1]

    def backward(self, f):
        f[:, 0, self.idxs] = self.cache[:, lb.OPPOSITE_IDXS[self.idxs]]

    def update_velocity(self, velocity_field):
        velocity_field[:, 0] = 0.0


class TopWallBoundary(WallBoundary):
    def __init__(self):
        self.idxs = [2, 5, 6]

    def forward(self, f):
        self.cache = f[1]

    def backward(self, f):
        f[0, :, [2, 5, 6]] = self.cache[:, lb.OPPOSITE_IDXS[self.idxs]].T

    def update_velocity(self, velocity_field):
        velocity_field[0, :] = 0.0


class BottomWallBoundary(WallBoundary):
    def __init__(self):
        self.idxs = [4, 7, 8]

    def forward(self, f):
        self.cache = f[-2]

    def backward(self, f):
        f[-1, :, self.idxs] = self.cache[:, lb.OPPOSITE_IDXS[self.idxs]].T

    def update_velocity(self, velocity_field):
        velocity_field[-1, :] = 0.0


class MovingWallBoundary(WallBoundary, ABC):
    def __init__(self, wall_velocity, cs=1/np.sqrt(3)):
        self.wall_velocity = wall_velocity
        self.cs = cs
        super().__init__()

    @abstractmethod
    def calculate_density(self, f):
        pass

    @abstractmethod
    def update_f(self, f, value):
        pass

    def backward(self, f):
        density = self.calculate_density(f)
        multiplier = 2 * (1/self.cs) ** 2
        momentum = multiplier * (lb.C @ self.wall_velocity) * (lb.W * density[:, None])
        momentum = momentum[:, lb.OPPOSITE_IDXS[self.idxs]]
        self.update_f(f, (self.cache[:, lb.OPPOSITE_IDXS[self.idxs]] - momentum).T)


class MovingTopWallBoundary(MovingWallBoundary, TopWallBoundary):
    def calculate_density(self, f):
        return lb.calculate_density(f[1])

    def update_velocity(self, velocity_field):
        velocity_field[0, :] = self.wall_velocity

    def update_f(self, f, value):
        f[0, :, self.idxs] = value


class MovingBottomWallBoundary(MovingWallBoundary, BottomWallBoundary):
    def calculate_density(self, f):
        return lb.calculate_density(f[-2])

    def update_velocity(self, velocity_field):
        velocity_field[-1, :] = self.wall_velocity

    def update_f(self, f, value):
        f[-1, :, self.idxs] = value


class HorizontalInletOutletBoundary(Boundary):
    def __init__(self, n, pressure_in=0.201, pressure_out=0.2, cs=1/np.sqrt(3)):
        self.p_in = pressure_in / cs**2
        self.p_out = pressure_out / cs**2
        self.cs = cs
        self.n = n

    def forward(self, f, feq, velocity):
        inlet_feq = lb.calculate_equilibrium_distribution(
                self.p_in * np.ones((self.n, 1), dtype=np.float32),
                velocity[:, None, -2]).squeeze()

        outlet_feq = lb.calculate_equilibrium_distribution(
                self.p_out * np.ones((self.n, 1), dtype=np.float32),
                velocity[:, None, 1]).squeeze()

        self.inlet_cache = inlet_feq + (f[:, -2] - feq[:, -2])
        self.outlet_cache = outlet_feq + (f[:, 1] - feq[:, 1])

    def backward(self, f):
        f[:, 0, [1, 5, 8]] = self.inlet_cache[:, [1, 5, 8]]
        f[:, -1, [3, 6, 7]] = self.outlet_cache[:, [3, 6, 7]]
