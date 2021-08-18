import os

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from simulations import shear_wave_simulation

rank = MPI.COMM_WORLD.Get_rank()


def viscosity_experiment(experiment_type):
    settings = {
        'nx': 50,
        'ny': 50,
        'eps': 0.01,
        'p0': 1.0,
        'n_steps': 3000,
        'save_every': None
        }

    ws = np.arange(0.1, 2.01, 0.1)

    simulated_viscosities = []
    analytical_viscosities = []
    for w in ws:
        simulated_viscosity, analytical_viscosity = shear_wave_simulation(
            **settings, w=w, experiment_type=experiment_type)
        simulated_viscosities.append(simulated_viscosity)
        analytical_viscosities.append(analytical_viscosity)

    if rank == 0:
        plt.cla()
        plt.scatter(ws, np.log(simulated_viscosities), marker='x')
        plt.scatter(ws, np.log(analytical_viscosities), marker='x')
        plt.xlabel('w')
        plt.ylabel('Log(Viscosity)')
        plt.legend(['Simulated', 'Analytical'])
        common_path = os.path.join('results', 'shear_decay')
        os.makedirs(common_path, exist_ok=True)
        path = os.path.join(common_path, f'viscosity_{experiment_type}.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)


viscosity_experiment('velocity')
viscosity_experiment('density')
