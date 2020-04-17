import numpy as np

def replace_zeros_with_mean(data):
    mu = sum(data)/np.count_nonzero(data, axis=0)
    non_zero = np.matrix(np.count_nonzero(data, axis=0))
    zero_mask = (data == 0)
    data[zero_mask] = mu[np.where(zero_mask)[1]]
    return data

