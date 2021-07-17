import numpy as np
import lb
import matplotlib.pyplot as plt
from mpi4py import MPI
import time


nx = 300
ny = 300
n_steps = 200

x, y = np.meshgrid(np.arange(nx), np.arange(ny))

# density = np.sin(2*np.pi/nx*x)
# velocity_field = np.zeros((2, ny, nx))

density = np.ones((ny, nx))
velocity_field = np.zeros((2, ny, nx))
velocity_field[1] = np.sin(2*np.pi/ny*y)

lbs = lb.LatticeBoltzmann(density, velocity_field)

prev_time = time.time()
for step in range(n_steps):
    lbs.stream()
    lbs.collide()

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     if not (step % 10):
    #         plt.cla()
    #         lbs.plot()
    #         plt.pause(0.001)

print(time.time() - prev_time)
