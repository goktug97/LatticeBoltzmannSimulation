import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import lb


parser = argparse.ArgumentParser(description='Shear Wave')
parser.add_argument('--nx', type=int, default=50, help='Lattice Width')
parser.add_argument('--ny', type=int, default=50, help='Lattice Height')
parser.add_argument('--wall-velocity', type=float, default=0.1)
parser.add_argument('--n-steps', type=int, default=100,
        help='Number of simulation steps')
parser.add_argument('--simulate', dest='simulate', action='store_true')
parser.set_defaults(simulate=False)

args = parser.parse_args()

rank = MPI.COMM_WORLD.Get_rank()

density = np.ones((args.ny, args.nx))
velocity_field = np.zeros((args.ny, args.nx, 2))

lbs = lb.LatticeBoltzmann(density, velocity_field)

x, y = np.meshgrid(range(args.nx), range(args.ny))
bottom_wall = y == args.ny - 1
top_wall = y == 0
wall_velocity = [0.0, args.wall_velocity]

prev_time = time.time()

lbs.add_boundary(lb.WallBoundary(bottom_wall))
lbs.add_boundary(lb.MovingWallBoundary(top_wall, wall_velocity))

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

lbs.gather_velocity_field()
if rank == 0:
    plt.streamplot(x, y, lbs.velocity_field[:, :, 1], lbs.velocity_field[:, :, 0])
    plt.show()


