from simulations import couette_flow_simulation

settings = {
    'nx': 50,
    'ny': 50,
    'wall_velocity': 0.1,
    'w': 0.3,
    'n_steps': 5000,
    'save_every': 200
    }

couette_flow_simulation(**settings)
