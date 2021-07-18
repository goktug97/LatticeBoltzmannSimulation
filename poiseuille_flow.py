import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import MovieWriter, FFMpegWriter
from mpi4py import MPI

import lb

nx = 150
ny = 100

re = 1000
inlet_velocity = 0.1
viscosity = inlet_velocity * nx / re
tau = 0.5 + viscosity / (1/3)
# assert 1/tau < 1.7

dt = re * viscosity / nx ** 2
n_steps = int(np.floor(8.0/dt))

# n_steps = 200

density = np.ones((ny, nx))
velocity_field = np.zeros((ny, nx, 2))

# velocity_field[0, :, 0] += 1

# f = np.ones((9, ny, nx))
# f += 0.01*np.random.randn(9,ny,nx)
# X, Y = np.meshgrid(range(nx), range(ny))
# print(2*np.pi*X/nx*32)
# f[1,:,:] += 2 * (1+0.2*np.cos(2*np.pi*X/nx*32))
# rho = np.sum(f,0)
# for i in range(9):
#     f[i,:,:] *= 100 / rho

lbs = lb.LatticeBoltzmann(density, velocity_field)

x, y = np.meshgrid(range(nx), range(ny))
bottom_wall = y == ny - 1
top_wall = y == 0

v = np.zeros((ny, 1, 2))
v[:, :, 1] += inlet_velocity

# Interesting Experiment
# v = np.zeros((2, ny, 1))
# v[1, :, 0] = inlet_velocity * (1 + np.sin(2*np.pi/ny*np.arange(ny)))

inlet_f = lb.calculate_equilibrium_distribution(np.ones((ny, 1)), v).squeeze()

# Place the cylinder a little bit off center to create a nice effect
cylinder = (x - nx/8)**2 + (y - ny/2.1)**2 < (ny/8)**2

# if MPI.COMM_WORLD.Get_rank() == 0:
#     fig, ax = plt.subplots()
#     moviewriter = FFMpegWriter()
#     moviewriter.setup(fig, 'animation.mp4', dpi=100)

prev_time = time.time()
for i in range(n_steps):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f'{i}\{n_steps}', end="\r")

    bottom_f = lbs.f[bottom_wall].copy()
    top_f = lbs.f[top_wall].copy()
    cylinder_f = lbs.f[cylinder].copy()
    outlet_f = lbs.f[:, -2, [3, 6, 7]].copy()

    lbs.stream()

    lbs.collide(tau)

    lbs.gather_f()

    # Cylinder
    lbs.velocity_field[cylinder] = 0
    lbs.f[cylinder, :] = cylinder_f[:, lb.OPPOSITE_IDXS]

    # Walls
    lbs.f[bottom_wall] = bottom_f[:, lb.OPPOSITE_IDXS]
    lbs.f[top_wall] = top_f[:, lb.OPPOSITE_IDXS]

    # Outlet
    lbs.f[:, -1, [3, 6, 7]] = outlet_f

    # Inlet
    lbs.f[:, 0] = inlet_f

    lbs.partial_update_f()

    # if not (i % 10):
    #     lbs.gather_velocity_field()
    #     if MPI.COMM_WORLD.Get_rank() == 0:
    #         ax.cla()
    #         lbs.plot(ax=ax)
    #         # moviewriter.grab_frame()
    #         plt.pause(0.001)


# if MPI.COMM_WORLD.Get_rank() == 0:
#     moviewriter.finish()

print(time.time() - prev_time)
