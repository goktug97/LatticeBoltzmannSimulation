import numpy as np

def _split(array, n, axis, rank):
    '''Split with padding.'''
    arrays = np.array_split(array, n, axis=axis)
    array = np.concatenate([np.take(arrays[rank-1], [-1], axis=axis),
        arrays[rank],
        np.take(arrays[(rank+1) % n], [0], axis=axis)], axis=axis)
    return array

