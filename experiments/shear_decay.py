from simulations import shear_wave_simulation

settings = {
    'nx': 50,
    'ny': 50,
    'w': 1.0,
    'eps': 0.01,
    'p0': 1.0,
    'n_steps': 3000,
    'save_every': 200
    }

shear_wave_simulation(**settings, experiment_type='velocity')
shear_wave_simulation(**settings, experiment_type='density')
