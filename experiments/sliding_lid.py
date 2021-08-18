from simulations import sliding_lid_simulation

settings = {
    'nx': 300,
    'ny': 300,
    're': 1000,
    'wall_velocity': 0.05,
    'n_steps': 100000,
    'save_every': 20000
    }

sliding_lid_simulation(**settings)
