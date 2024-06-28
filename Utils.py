import numpy as np
from Variable import Variable

def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

def numerical_diff(fx, x: Variable, eps=1e-4):
    x_l = Variable(as_array(x.data-eps))
    x_r = Variable(as_array(x.data+eps))
    y_l = fx(x_l)
    y_r = fx(x_r)
    return (y_r.data - y_l.data) / (2*eps)
