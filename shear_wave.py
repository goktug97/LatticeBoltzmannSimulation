import argparse
import time

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

import lb


parser = argparse.ArgumentParser(description='Shear Wave')
parser.add_argument('--nx', type=int, default=300, help='Lattice Width')
parser.add_argument('--ny', type=int, default=300, help='Lattice Height')
parser.add_argument('--n-steps', type=int, default=100,
        help='Number of simulation steps')
parser.add_argument('--simulate', dest='simulate', action='store_true')
parser.set_defaults(simulate=False)

args = parser.parse_args()

rank = MPI.COMM_WORLD.Get_rank()

x, y = np.meshgrid(np.arange(args.nx), np.arange(args.ny))

# density = np.sin(2*np.pi/args.nx*x)
# velocity_field = np.zeros((args.ny, args.nx, 2))

density = np.ones((args.ny, args.nx), dtype=np.float32)
velocity_field = np.zeros((args.ny, args.nx, 2), dtype=np.float32)
velocity_field[:, :, 1] = np.sin(2*np.pi/args.ny*y)

lbs = lb.LatticeBoltzmann(density, velocity_field)

prev_time = time.time()

for step in range(args.n_steps):
    if rank == 0:
        print(f'{step+1}\{args.n_steps}', end="\r")

    lbs.step()

    if args.simulate:
        if not (step % 10):
            lbs.gather_velocity_field()
            if rank == 0:
                plt.cla()
                lbs.plot()
                plt.pause(0.001)

print(f'Core {rank}: Simulation Time: {time.time() - prev_time}')

if rank == 0:
    lbs.gather_velocity_field()
    plt.streamplot(x, y, lbs.velocity_field[:, :, 1], lbs.velocity_field[:, :, 0])
    plt.show()

