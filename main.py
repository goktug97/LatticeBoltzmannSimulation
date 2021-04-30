import numpy as np

nx = 15
ny = 10

F = np.random.normal(0, 0.1, (9, nx, ny))

c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
              [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T

for i in range(9):
    F[i] = np.roll(F[i], c[i], axis=(0, 1))

density = np.sum(F, axis=0)
velocity_field = np.dot(F.T, c).T/density
print(velocity_field.shape)
