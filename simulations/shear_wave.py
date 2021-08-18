import argparse
import os
from typing import Optional

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

import lb


def shear_wave_simulation(nx: int = 50, ny: int = 50, w: float = 1.0, eps: float = 0.01,
                          n_steps: int = 3000, simulate: bool = False,
                          experiment_type: str = 'velocity', p0: float = 1.0,
                          save_every: Optional[int] = None,
                          output_dir: str = 'results'):
    rank = MPI.COMM_WORLD.Get_rank()

    x, y = np.meshgrid(np.arange(nx), np.arange(ny))

    if experiment_type == 'velocity':
        density = np.ones((ny, nx), dtype=np.float32)
        velocity_field = np.zeros((ny, nx, 2), dtype=np.float32)
        velocity_field[:, :, 1] = eps * np.sin(2*np.pi/ny*y)
    else:
        density = p0 + eps * np.sin(2*np.pi/nx*x)
        velocity_field = np.zeros((ny, nx, 2), dtype=np.float32)

    lbs = lb.LatticeBoltzmann(density, velocity_field)

    tau = 1 / w

    q = []

    common_path = os.path.join(output_dir, 'shear_decay')
    if rank == 0:
        if save_every is not None:
            path = os.path.join(common_path, f'decay_{experiment_type}')
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

        if simulate:
            if not (step % 100):
                lbs.gather_velocity_field()
                if rank == 0:
                    axes[1].cla()
                    lbs.plot(ax=axes[1])
                    figs[1].canvas.draw()
                    figs[1].canvas.flush_events()
                    figs[1].show()

        if experiment_type == 'velocity':
            q.append(np.max(np.abs(lbs.velocity_field[:, :, 1])))
        else:
            q.append(np.max(np.abs(lbs.density - p0)))

        if save_every is not None and (not (step % save_every) or step == n_steps - 1):
            if experiment_type == 'velocity':
                lbs.gather_velocity_field()
                if rank == 0:
                    axes[0].cla()
                    axes[0].set_ylim([-eps, eps])
                    axes[0].plot(np.arange(ny), lbs.velocity_field[:, nx//2, 1])
                    axes[0].set_xlabel('y')
                    axes[0].set_ylabel('velocity')
                    save_path = os.path.join(path, f'velocity_decay_{step}.png')
                    figs[0].savefig(save_path, bbox_inches='tight', pad_inches=0)
            else:
                lbs.gather_density()
                if rank == 0:
                    axes[0].cla()
                    axes[0].set_ylim([-eps + p0, eps + p0])
                    axes[0].plot(np.arange(nx), lbs.density[ny//2, :])
                    axes[0].set_xlabel('x')
                    axes[0].set_ylabel('density')
                    save_path = os.path.join(path, f'density_decay_{step}.png')
                    figs[0].savefig(save_path, bbox_inches='tight', pad_inches=0)

    def decay_perturbation(t, viscosity):
        size = ny if experiment_type == 'velocity' else nx
        return eps * np.exp(-viscosity * (2*np.pi/size)**2 * t)

    if experiment_type == 'density':
        q = np.array(q)
        x = argrelextrema(q, np.greater)[0]
        q = q[x]
    else:
        x = np.arange(n_steps)

    simulated_viscosity = curve_fit(decay_perturbation, xdata=x, ydata=q)[0][0]
    analytical_viscosity = (1/3) * ((1/w) - 0.5)

    for fig in figs:
        plt.close(fig)

    return simulated_viscosity, analytical_viscosity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shear Wave')
    parser.add_argument('--nx', type=int, default=50, help='Lattice Width')
    parser.add_argument('--ny', type=int, default=50, help='Lattice Height')
    parser.add_argument('--w', type=float, default=1.0, help='Omega')
    parser.add_argument('--eps', type=float, default=0.01, help='Magnitude')
    parser.add_argument('--n-steps', type=int, default=2000,
                        help='Number of simulation steps')
    parser.add_argument('--simulate', dest='simulate', action='store_true')
    parser.add_argument('--experiment-type', type=str, default='velocity',
                        choices=['velocity', 'density'])
    parser.add_argument('--p0', type=float, default=1.0, help='Density offset')
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.set_defaults(simulate=False)
    args = parser.parse_args()

    simulated_viscosity, analytical_viscosity = shear_wave_simulation(**vars(args))
