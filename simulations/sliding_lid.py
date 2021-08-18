from typing import Optional
import time
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import lb


def sliding_lid_simulation(nx: int = 300, ny: int = 300, re: float = 1000,
                           wall_velocity: float = 0.05, cs: float = 1/np.sqrt(3),
                           simulate: bool = False, save_every: Optional[int] = None,
                           n_steps: int = 100000, output_dir: str = 'results'):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_every is not None:
        if int(wall_velocity) == wall_velocity:
            str_u = str(int(wall_velocity))
        else:
            str_u = str(wall_velocity).replace('.', '')
        if int(re) == re:
            str_re = str(int(re))
        else:
            str_re = str(re).replace('.', '')
        path = os.path.join(output_dir, f'sliding_lid_{nx}_{ny}_{str_re}_{str_u}')
        if rank == 0:
            os.makedirs(path, exist_ok=True)

    if simulate:
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        figs, axes = [fig, fig2], [ax, ax2]
    else:
        fig, ax = plt.subplots()
        figs, axes = [fig], [ax]

    viscosity = wall_velocity * nx / re
    tau = 0.5 + viscosity / (1/3)
    # assert 1/tau <= 1.7

    density = np.ones((ny, nx))
    velocity_field = np.zeros((ny, nx, 2))

    lbs = lb.LatticeBoltzmann(density, velocity_field)

    x, y = np.meshgrid(range(nx), range(ny))
    wall_velocity = [0.0, wall_velocity]

    lbs.add_boundary(lb.BottomWallBoundary())
    lbs.add_boundary(lb.MovingTopWallBoundary(wall_velocity, cs))
    lbs.add_boundary(lb.RightWallBoundary())
    lbs.add_boundary(lb.LeftWallBoundary())

    prev_time = time.time()

    for step in range(n_steps):
        if rank == 0:
            print(f'{step+1}\\{n_steps}', end="\r")

        lbs.stream_and_collide(tau)

        if save_every is not None and (not (step % save_every) or (step == n_steps-1)):
            axes[0].cla()
            axes[0].set_xlim([0, nx])
            axes[0].set_xlim([0, ny])
            lbs.plot(ax=axes[0])
            lbs.streamplot(ax=axes[0])
            save_path = os.path.join(path, f'sliding_lid_{step}.png')
            figs[0].savefig(save_path, bbox_inches='tight', pad_inches=0)

        if simulate:
            if not (step % 100):
                lbs.gather_velocity_field()
                if rank == 0:
                    axes[1].cla()
                    lbs.plot(ax=axes[1])
                    figs[1].canvas.draw()
                    figs[1].canvas.flush_events()
                    figs[1].show()

    mlups = nx * ny * n_steps / (time.time() - prev_time) / 1e6

    for fig in figs:
        plt.close(fig)

    return mlups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sliding Lid')
    parser.add_argument('--nx', type=int, default=300, help='Lattice Width')
    parser.add_argument('--ny', type=int, default=300, help='Lattice Height')
    parser.add_argument('--re', type=float, default=1000, help='Reynolds Number')
    parser.add_argument('--cs', type=float, default=1/np.sqrt(3), help='Speed of sound')
    parser.add_argument('--wall-velocity', type=float, default=0.05)
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--simulate', dest='simulate', action='store_true')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--save-every', type=int, default=None)
    parser.set_defaults(simulate=False)
    args = parser.parse_args()

    sliding_lid_simulation(**vars(args))
