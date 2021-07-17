import numpy as np
import lb
import matplotlib.pyplot as plt
from mpi4py import MPI


nx = 50
ny = 50
n_steps = 20000

density = np.ones((ny, nx))
velocity_field = np.zeros((2, ny, nx))

lbs = lb.LatticeBoltzmann(density, velocity_field)

x, y = np.meshgrid(range(nx), range(ny))
bottom_wall = y == ny - 1
top_wall = y == 0
wall_velocity = [0.0, 0.1]

for step in range(n_steps):
    lbs.stream()

    bottom_f = lbs.f[:, bottom_wall].copy()
    top_f = lbs.f[:, top_wall].copy()

    lbs.collide()

    lbs.f[:, bottom_wall] = bottom_f[lb.OPPOSITE_IDXS]
    momentum = 6 * (lb.C @ wall_velocity)[:, None] * (lb.W[:, None] * density[None, top_wall])
    lbs.f[:, top_wall] = top_f[lb.OPPOSITE_IDXS, :] + momentum

    if MPI.COMM_WORLD.Get_rank() == 0:
        if not (step % 10):
            plt.cla()
            lbs.plot()
            plt.pause(0.001)
