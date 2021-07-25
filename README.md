Lattice Boltzmann Simulation
============================

<p align="middle">
    <img src="/animation.gif" width="50%" height="100%"/>
</p>


## Requirements
```bash
numpy
matplotlib
mpi4py
```

## Usage
```bash
git clone https://github.com/goktug97/LatticeBoltzmannSimulation
cd LatticeBoltzmannSimulation

# Single thread
PYTHONPATH="$(pwd):$PYTHONPATH" python simulations/poiseuille_flow.py --simulate

# Parallel
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python simulations/poiseuille_flow.py --simulate
```
