import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import lb


parser = argparse.ArgumentParser(description='Poiseuille Flow')
parser.add_argument('--nx', type=int, default=300, help='Lattice Width')
parser.add_argument('--ny', type=int, default=300, help='Lattice Height')
parser.add_argument('--re', type=float, default=100, help='Reynolds Number')
parser.add_argument('--wall-velocity', type=float, default=0.1)
parser.add_argument('--n-steps', type=int, default=None,
        help='Number of simulation steps')
parser.add_argument('--simulate', dest='simulate', action='store_true')
parser.set_defaults(simulate=False)
args = parser.parse_args()

rank = MPI.COMM_WORLD.Get_rank()

viscosity = args.wall_velocity * args.nx / args.re
tau = 0.5 + viscosity / (1/3)
# assert 1/tau < 1.7

if args.n_steps is None:
    dt = args.re * viscosity / args.nx ** 2
    n_steps = int(np.floor(8.0/dt))
else:
    n_steps = args.n_steps

density = np.ones((args.ny, args.nx))
velocity_field = np.zeros((args.ny, args.nx, 2))

lbs = lb.LatticeBoltzmann(density, velocity_field)

x, y = np.meshgrid(range(args.nx), range(args.ny))
bottom_wall = y == args.ny - 1
top_wall = y == 0
right_wall = x == args.nx - 1
left_wall = x == 0
wall_velocity = [0.0, args.wall_velocity]


lbs.add_boundary(lb.WallBoundary(bottom_wall))
lbs.add_boundary(lb.WallBoundary(left_wall))
lbs.add_boundary(lb.WallBoundary(right_wall))
lbs.add_boundary(lb.MovingWallBoundary(top_wall, wall_velocity))

prev_time = time.time()

for step in range(n_steps):
    if rank == 0:
        print(f'{step+1}\{n_steps}', end="\r")

    lbs.step(tau)

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
