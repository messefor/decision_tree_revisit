"""Utils."""
import numpy as np

# sigmoid function
def sigmoid(x, shift=0, scale=1):
    """Return sigmoid."""
    return 1 / (1 + np.exp(-(x - shift) / scale))

def rand_range(size, r_min, r_max):
    """ """
    return np.random.rand(size) * (r_max  - r_min) + r_min
