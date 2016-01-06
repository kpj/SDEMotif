"""
Useful functions and more
"""

import numpy as np
import scipy.stats as scis


def get_nonconst_data(i, j, data):
    """ Return data along some axis and make sure it's not constant
    """
    xs, ys = data[:,i], data[:,j]

    # fix series with stdev == 0
    xs[0] += 1e-10
    ys[0] += 1e-10

    return xs, ys

def compute_correlation_matrix(data):
    """ Compute correlation matrix of given data points
    """
    dim = data.shape[1]

    mat = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            xs, ys = get_nonconst_data(i, j, data)
            cc, pval = scis.pearsonr(xs, ys)
            mat[i, j] = cc

    return mat
