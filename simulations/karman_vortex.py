import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import lb


def karman_vortex_simulation(nx=100, ny=50, re=1000, inlet_velocity=0.1,
                             n_steps=5000, text='cylinder', sim_type='normal'):
    rank = MPI.COMM_WORLD.Get_rank()

    viscosity = args.inlet_velocity * args.nx / args.re
    tau = 0.5 + viscosity / (1/3)

    density = np.ones((args.ny, args.nx))
    velocity_field = np.zeros((args.ny, args.nx, 2))

    lbs = lb.LatticeBoltzmann(density, velocity_field)

    x, y = np.meshgrid(range(args.nx), range(args.ny))

    v = np.zeros((args.ny, 1, 2))
    if args.sim_type == 'normal':
        v[:, :, 1] += args.inlet_velocity
    else:
        v[:, 0, 1] = args.inlet_velocity * (1 + np.sin(2*np.pi/args.ny*np.arange(args.ny)))

    inlet_f = lb.calculate_equilibrium_distribution(
        np.ones((args.ny, 1), dtype=np.float32), v).squeeze()

    lbs.add_boundary(lb.BottomWallBoundary())
    lbs.add_boundary(lb.TopWallBoundary())

    if args.text == 'cylinder':
        # Place the cylinder a little bit off center to create the karman vortex
        cylinder = (x - args.nx/8)**2 + (y - args.ny/2.1)**2 < (args.ny/8)**2
        lbs.add_boundary(lb.RigidObject(cylinder))
    else:
        import cv2
        mask = np.zeros((args.ny, args.nx), np.uint8)
        cv2.putText(mask, args.text, (10, args.ny//2+3),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, 255,
                    lineType=cv2.LINE_8)
        lbs.add_boundary(lb.RigidObject(mask == 255))

    for step in range(args.n_steps):
        if rank == 0:
            print(f'{step+1}\\{args.n_steps}', end="\r")

        lbs.stream_and_collide(tau)
        lbs.f[:, -1, [1, 5, 8]] = inlet_f[:, [1, 5, 8]]

        if not (step % 10):
            lbs.gather_velocity_field()
            if rank == 0:
                plt.cla()
                lbs.plot()
                plt.pause(0.001)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Karman Vortex')
    parser.add_argument('--nx', type=int, default=100, help='Lattice Width')
    parser.add_argument('--ny', type=int, default=50, help='Lattice Height')
    parser.add_argument('--re', type=float, default=1000, help='Reynolds Number')
    parser.add_argument('--inlet-velocity', type=float, default=0.1)
    parser.add_argument('--n-steps', type=int, default=10000,
                        help='Number of simulation steps')
    parser.add_argument('--text', type=str, default='cylinder')
    parser.add_argument('--sim-type', choices=['normal', 'interesting'],
                        default='normal')

    args = parser.parse_args()

    karman_vortex_simulation(**vars(args))
