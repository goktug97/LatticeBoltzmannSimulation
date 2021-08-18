import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import lb


def poiseuille_flow_simulation(nx: int = 100, ny: int = 50, w: float = 0.3,
                               p_in: float = 0.31, p_out: float = 0.3,
                               cs: float = 1/np.sqrt(3), simulate: bool = False,
                               n_steps: int = 30000, save_every: int = 1000,
                               output_dir: str = 'results'):
    rank = MPI.COMM_WORLD.Get_rank()

    tau = 1 / w

    viscosity = 1/3 * (tau - 0.5)

    n_steps = n_steps

    density = np.ones((ny, nx))
    velocity_field = np.zeros((ny, nx, 2))

    lbs = lb.LatticeBoltzmann(density, velocity_field)

    y = np.arange(ny)

    lbs.add_boundary(lb.TopWallBoundary())
    lbs.add_boundary(lb.BottomWallBoundary())
    lbs.add_boundary(lb.HorizontalInletOutletBoundary(
        ny, p_in, p_out, cs))

    if rank == 0:
        path = os.path.join(output_dir, 'pousielle_flow')
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

        lbs.stream_and_collide(tau)

        if not (step % save_every) or (step == n_steps-1):
            lbs.gather_density()
            lbs.gather_velocity_field()
            if rank == 0:
                axes[0].cla()
                dynamic_viscosity = (lbs.density[:, nx//2] * viscosity)
                partial_derivative = (p_out - p_in) / nx
                analytical = (-0.5 * partial_derivative * y * (ny - 1 - y)) / dynamic_viscosity
                # axes[0].set_xlim([0, np.max(analytical) + 0.001])
                for boundary in lbs.boundaries:
                    boundary.update_velocity(lbs.velocity_field)
                axes[0].plot(lbs.velocity_field[:, nx//2, 1], y)
                axes[0].plot(analytical, y)
                axes[0].set_ylabel('y')
                axes[0].set_xlabel('velocity')
                axes[0].legend(['Simulated', 'Analytical'])
                save_path = os.path.join(path, f'pousielle_flow_{step}')
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
    parser = argparse.ArgumentParser(description='Poiseuille Flow')
    parser.add_argument('--nx', type=int, default=100, help='Lattice width')
    parser.add_argument('--ny', type=int, default=50, help='Lattice height')
    parser.add_argument('--w', type=float, default=0.3, help='Omega')
    parser.add_argument('--n-steps', type=int, default=30000)
    parser.add_argument('--simulate', dest='simulate', action='store_true')
    parser.add_argument('--p-out', type=float, default=0.3, help='Input pressure')
    parser.add_argument('--p-in', type=float, default=0.3005, help='Output pressure')
    parser.add_argument('--cs', type=float, default=1/np.sqrt(3), help='Speed of sound')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.set_defaults(simulate=False)
    args = parser.parse_args()

    poiseuille_flow_simulation(**vars(args))
