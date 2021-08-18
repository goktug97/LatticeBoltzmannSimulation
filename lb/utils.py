import numpy as np

C = np.ascontiguousarray(
        np.array([[0, 0, 1, 0, -1, 1, 1, -1, -1], # y
                  [0, 1, 0, -1, 0, 1, -1, -1, 1]]).T) # x
C.setflags(write=False)

W = np.ascontiguousarray(
        np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T,
        dtype=np.float32)
W.setflags(write=False)

OPPOSITE_IDXS = np.ascontiguousarray([0, 3, 4, 1, 2, 7, 8, 5, 6])

OPPOSITE_IDXS.setflags(write=False)


def calculate_density(f):
    density = np.sum(f, axis=-1)
    return density


def calculate_velocity_field(f, density):
    velocity_field = np.dot(f, C) / (density[:, :, None] + np.finfo(np.float32).eps)
    return velocity_field


def calculate_equilibrium_distribution(density, velocity_field):
    c_dot_vf = (velocity_field[:, :, :, None] * C.T[None, None])
    c_dot_vf = np.sum(c_dot_vf, axis=2)
    vf_norm_square = np.sum(velocity_field**2, axis=2)[:, :, None]
    feq = W * (density[:, :, None] * (1 + 3 * c_dot_vf + 4.5*c_dot_vf**2 - 1.5*vf_norm_square))
    return feq
