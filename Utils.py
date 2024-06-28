import numpy as np

def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x
