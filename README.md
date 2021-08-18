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
All commands must be run in the project root.
```bash
git clone https://github.com/goktug97/LatticeBoltzmannSimulation
pip install -r requirements.txt

# Single thread
PYTHONPATH="$(pwd):$PYTHONPATH" python simulations/karman_vortex.py

# Parallel
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python simulations/karman_vortex.py
```

Check `simulations` folder for more examples.

```python
import numpy as np
import matplotlib.pyplot as plt

import lb


nx = 50
ny = 50
n_steps = 3000
w = 0.3

x, y = np.meshgrid(np.arange(nx), np.arange(ny))

density = np.ones((ny, nx), dtype=np.float32)
velocity_field = np.zeros((ny, nx, 2), dtype=np.float32)
velocity_field[:, :, 1] = np.sin(2*np.pi/ny*y)

lbs = lb.LatticeBoltzmann(density, velocity_field)

for step in range(n_steps):
    print(f'{step+1}\\{n_steps}', end="\r")

    lbs.stream_and_collide(tau=1/w)

    if not step % 100:
        plt.cla()
        lbs.plot()
        plt.pause(0.001)
```

## Experiments
All commands must be run in the project root.
All results are saved into `results` folder.

### Shear Wave Decay Experiment
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python experiments/shear_decay.py

```
#### Velocity Decay
||||
|:-:|:-:|:-:|
![](plots/shear_decay_velocity/velocity_decay_0.png)Step 0 | ![](plots/shear_decay_velocity/velocity_decay_200.png)Step 200 | ![](plots/shear_decay_velocity/velocity_decay_400.png)Step 400
![](plots/shear_decay_velocity/velocity_decay_600.png)Step 600 | ![](plots/shear_decay_velocity/velocity_decay_800.png)Step 800 | ![](plots/shear_decay_velocity/velocity_decay_1000.png)Step 1000
![](plots/shear_decay_velocity/velocity_decay_1200.png)Step 1200 | ![](plots/shear_decay_velocity/velocity_decay_1400.png)Step 1400 | ![](plots/shear_decay_velocity/velocity_decay_1600.png)Step 1600
![](plots/shear_decay_velocity/velocity_decay_1800.png)Step 1800 | ![](plots/shear_decay_velocity/velocity_decay_2000.png)Step 2000 | ![](plots/shear_decay_velocity/velocity_decay_2999.png)Step 2999

#### Density Decay
||||
|:-:|:-:|:-:|
![](plots/shear_decay_density/density_decay_0.png)Step 0 | ![](plots/shear_decay_density/density_decay_200.png)Step 200 | ![](plots/shear_decay_density/density_decay_400.png)Step 400
![](plots/shear_decay_density/density_decay_600.png)Step 600 | ![](plots/shear_decay_density/density_decay_800.png)Step 800 | ![](plots/shear_decay_density/density_decay_1000.png)Step 1000
![](plots/shear_decay_density/density_decay_1200.png)Step 1200 | ![](plots/shear_decay_density/density_decay_1400.png)Step 1400 | ![](plots/shear_decay_density/density_decay_1600.png)Step 1600
![](plots/shear_decay_density/density_decay_1800.png)Step 1800 | ![](plots/shear_decay_density/density_decay_2000.png)Step 2000 | ![](plots/shear_decay_density/density_decay_2999.png)Step 2999

### Shear Wave Simulated vs Analytical Viscosity
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python experiments/shear_wave_viscosity.py
```
| Density                         | Velocity                         |
:--------------------------------:|:---------------------------------:
![](plots/viscosity_density.png)  | ![](plots/viscosity_velocity.png)

### Couette Flow
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python experiments/couette_flow.py
```
||||
|:-:|:-:|:-:|
![](plots/couette_flow/couette_flow_0.png)Step 0 | ![](plots/couette_flow/couette_flow_200.png)Step 200 | ![](plots/couette_flow/couette_flow_400.png)Step 400
![](plots/couette_flow/couette_flow_600.png)Step 600 | ![](plots/couette_flow/couette_flow_800.png)Step 800 | ![](plots/couette_flow/couette_flow_1000.png)Step 1000
![](plots/couette_flow/couette_flow_1200.png)Step 1200 | ![](plots/couette_flow/couette_flow_1400.png)Step 1400 | ![](plots/couette_flow/couette_flow_1600.png)Step 1600

### Poiseuille Flow
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python experiments/poiseuille_flow.py
```
||||
|:-:|:-:|:-:|
![](plots/pousielle_flow/pousielle_flow_0.png)Step 0 | ![](plots/pousielle_flow/pousielle_flow_400.png)Step 400 | ![](plots/pousielle_flow/pousielle_flow_800.png)Step 800
![](plots/pousielle_flow/pousielle_flow_1200.png)Step 1200 | ![](plots/pousielle_flow/pousielle_flow_1600.png)Step 1600 | ![](plots/pousielle_flow/pousielle_flow_2000.png)Step 2000
![](plots/pousielle_flow/pousielle_flow_2400.png)Step 2400 | ![](plots/pousielle_flow/pousielle_flow_3200.png)Step 3200 | ![](plots/pousielle_flow/pousielle_flow_3999.png)Step 3999

### Lid-driven Cavity
```bash
PYTHONPATH="$(pwd):$PYTHONPATH" mpirun -np 2 python experiments/sliding_lid.py
```
||||
|:-:|:-:|:-:|
![](plots/sliding_lid_300_300_1000_005/sliding_lid_0.png)Step 0 | ![](plots/sliding_lid_300_300_1000_005/sliding_lid_20000.png)Step 20000 | ![](plots/sliding_lid_300_300_1000_005/sliding_lid_40000.png)Step 40000
![](plots/sliding_lid_300_300_1000_005/sliding_lid_60000.png)Step 60000 | ![](plots/sliding_lid_300_300_1000_005/sliding_lid_80000.png)Step 80000 | ![](plots/sliding_lid_300_300_1000_005/sliding_lid_99999.png)Step 99999
