from simulations import poiseuille_flow_simulation

settings = {
    'nx': 100,
    'ny': 50,
    'w': 0.3,
    'p_in': 3.005,
    'p_out': 3.0,
    'n_steps': 4000,
    'save_every': 200
    }

poiseuille_flow_simulation(**settings)
