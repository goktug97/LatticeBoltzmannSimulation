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

Check `simulations` folder.

```python
import numpy as np
import matplotlib.pyplot as plt

import lb


nx = 300
ny = 300
n_steps = 20

x, y = np.meshgrid(np.arange(nx), np.arange(ny))

density = np.ones((ny, nx), dtype=np.float32)
velocity_field = np.zeros((ny, nx, 2), dtype=np.float32)
velocity_field[:, :, 1] = np.sin(2*np.pi/ny*y)

lbs = lb.LatticeBoltzmann(density, velocity_field)

fig, axes = plt.subplots(1, 2)
for step in range(n_steps):
    print(f'{step+1}\{n_steps}', end="\r")

    lbs.stream_and_collide()

    axes[0].cla()
    lbs.plot(axes[0])
    plt.pause(0.001)

lbs.streamplot(axes[1])
plt.show()
```
