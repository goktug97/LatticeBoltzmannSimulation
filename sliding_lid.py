import numpy as np
import lb
import matplotlib.pyplot as plt
from mpi4py import MPI


nx = 300
ny = 300

re = 100
wall_velocity = 0.1
viscosity = wall_velocity * nx / re
tau = 0.5 + viscosity / (1/3)
print(1/tau)
assert 1/tau < 1.7

dt = re * viscosity / nx ** 2
n_steps = int(np.floor(8.0/dt))

density = np.ones((ny, nx))
velocity_field = np.zeros((2, ny, nx))

lbs = lb.LatticeBoltzmann(density, velocity_field)

x, y = np.meshgrid(range(nx), range(ny))
bottom_wall = y == ny - 1
top_wall = y == 0
right_wall = x == nx - 1
left_wall = x == 0
wall_velocity = [0.0, wall_velocity]

for step in range(n_steps):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'{step}\{n_steps}', end="\r")

    bottom_f = lbs.f[:, bottom_wall].copy()
    top_f = lbs.f[:, top_wall].copy()

    left_f = lbs.f[:, left_wall].copy()
    right_f = lbs.f[:, right_wall].copy()

    lbs.stream()

    lbs.f[:, left_wall] = left_f[lb.OPPOSITE_IDXS]
    lbs.f[:, right_wall] = right_f[lb.OPPOSITE_IDXS]
    lbs.f[:, bottom_wall] = bottom_f[lb.OPPOSITE_IDXS]

    momentum = 6 * (lb.C @ wall_velocity)[:, None] * (lb.W[:, None] * density[None, top_wall])

    # This is the correct equation which worked kind of wrong for me?
    # lbs.f[:, top_wall] = top_f[lb.OPPOSITE_IDXS, :] - momentum

    # This is the fix?
    lbs.f[:, top_wall] = top_f[lb.OPPOSITE_IDXS, :] + momentum

    lbs.collide(tau)

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     if not (step % 10):
    #         plt.cla()
    #         lbs.plot()
    #         plt.pause(0.001)

if MPI.COMM_WORLD.Get_rank() == 0:
    plt.streamplot(x, y, lbs.velocity_field[1], lbs.velocity_field[0])
    plt.show()
