import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import MovieWriter, FFMpegWriter
from mpi4py import MPI

import lb

parser = argparse.ArgumentParser(description='Poiseuille Flow')
parser.add_argument('--nx', type=int, default=200, help='Lattice Width')
parser.add_argument('--ny', type=int, default=100, help='Lattice Height')
parser.add_argument('--re', type=float, default=1000, help='Reynolds Number')
parser.add_argument('--inlet-velocity', type=float, default=0.1)
parser.add_argument('--n-steps', type=int, default=None,
        help='Number of simulation steps')
parser.add_argument('--sim-type', choices=['normal', 'interesting'],
        default='normal')
parser.add_argument('--no-cylinder', dest='cylinder', action='store_false')
parser.add_argument('--simulate', dest='simulate', action='store_true')
parser.add_argument('--save-simulation', type=str, default=None)
parser.set_defaults(simulate=False)
parser.set_defaults(cylinder=True)

args = parser.parse_args()

rank = MPI.COMM_WORLD.Get_rank()

viscosity = args.inlet_velocity * args.nx / args.re
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

v = np.zeros((args.ny, 1, 2))
if args.sim_type == 'normal':
    v[:, :, 1] += args.inlet_velocity
else:
    v[:, 0, 1] = args.inlet_velocity * (1 + np.sin(2*np.pi/args.ny*np.arange(args.ny)))

if rank == 0:
    if args.simulate:
        fig, ax = plt.subplots()
        if args.save_simulation is not None:
            path = Path(args.save_simulation).with_suffix('.mp4')
            moviewriter = FFMpegWriter()
            moviewriter.setup(fig, path, dpi=100)

bottom_wall = y == args.ny - 1
top_wall = y == 0
lbs.add_boundary(lb.WallBoundary(bottom_wall))
lbs.add_boundary(lb.WallBoundary(top_wall))
lbs.add_boundary(lb.HorizontalInletOutletBoundary(args.ny, v))

if args.cylinder:
    # Place the cylinder a little bit off center to create a nice effect
    cylinder = (x - args.nx/8)**2 + (y - args.ny/2.1)**2 < (args.ny/8)**2
    lbs.add_boundary(lb.WallBoundary(cylinder))

prev_time = time.time()

for step in range(n_steps):
    if rank == 0:
        print(f'{step+1}\{n_steps}', end="\r")

    lbs.stream_and_collide(tau)

    if args.simulate:
        if not (step % 10):
            lbs.gather_velocity_field()
            if rank == 0:
                ax.cla()
                lbs.plot(ax=ax)
                if args.save_simulation is not None:
                    moviewriter.grab_frame()
                else:
                    plt.pause(0.001)

print(f'Core {rank}: Simulation Time: {time.time() - prev_time}')

lbs.gather_velocity_field()
if rank == 0:
    fig, ax = plt.subplots()
    ax.streamplot(x, y, lbs.velocity_field[:, :, 1], lbs.velocity_field[:, :, 0])
    plt.show()

    if args.simulate and args.save_simulation is not None:
        moviewriter.finish()

