import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import lb


def couette_flow_simulation(nx: int = 50, ny: int = 50, wall_velocity: float = 0.1,
                            w: float = 0.3, simulate: bool = False,
                            n_steps: int = 30000, save_every: int = 1000,
                            output_dir: str = 'results'):
    rank = MPI.COMM_WORLD.Get_rank()

    tau = 1 / w

    density = np.ones((ny, nx))
    velocity_field = np.zeros((ny, nx, 2))

    lbs = lb.LatticeBoltzmann(density, velocity_field)

    y = np.arange(ny)
    wall_velocity = [0.0, wall_velocity]

    analytical = (ny-1 - y) / (ny-1) * wall_velocity[1]

    lbs.add_boundary(lb.BottomWallBoundary())
    lbs.add_boundary(lb.MovingTopWallBoundary(wall_velocity))

    if rank == 0:
        path = os.path.join(output_dir, 'couette_flow')
        os.makedirs(path, exist_ok=True)

    if simulate:
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        figs, axes = [fig, fig2], [ax, ax2]
    else:
        fig, ax = plt.subplots()
        figs, axes = [fig], [ax]

    for step in range(n_steps):
        if rank == 0:
            print(f'{step+1}\\{n_steps}', end="\r")

        lbs.stream_and_collide(tau=tau)

        if not (step % save_every) or step == n_steps - 1:
            lbs.gather_velocity_field()
            if rank == 0:
                axes[0].cla()
                axes[0].set_xlim([-0.01, wall_velocity[1]])
                axes[0].axhline(0.0, color='k')
                axes[0].axhline(ny-1, color='r')
                for boundary in lbs.boundaries:
                    boundary.update_velocity(lbs.velocity_field)
                axes[0].plot(lbs.velocity_field[:, nx//2, 1], y)
                axes[0].plot(analytical, y)
                axes[0].set_ylabel('y')
                axes[0].set_xlabel('velocity')
                axes[0].legend(['Moving Wall', 'Rigid Wall',
                                'Simulated Velocity', 'Analytical Velocity'])
                save_path = os.path.join(path, f'couette_flow_{step}')
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

    for fig in figs:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Couette Flow')
    parser.add_argument('--nx', type=int, default=50, help='Lattice Width')
    parser.add_argument('--ny', type=int, default=50, help='Lattice Height')
    parser.add_argument('--wall-velocity', type=float, default=0.1)
    parser.add_argument('--n-steps', type=int, default=30000,
                        help='Number of simulation steps')
    parser.add_argument('--w', type=float, default=0.3, help='Omega')
    parser.add_argument('--simulate', dest='simulate', action='store_true')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.set_defaults(simulate=False)

    args = parser.parse_args()

    couette_flow_simulation(**vars(args))
