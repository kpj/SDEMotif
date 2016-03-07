"""
Useful functions and more
"""

import os

import numpy as np
import scipy.stats as scis


def get_correlation(xs, ys):
    """ Compute correlation and handle equal series
    """
    try:
        cc, pval = scis.pearsonr(xs, ys)
    except FloatingPointError:
        cc, pval = 0, 0
    return cc

def extract(i, j, data):
    """ Extract data vectors
    """
    return data[:,i], data[:,j]

def compute_correlation_matrix(data):
    """ Compute correlation matrix of given data points
    """
    dim = data.shape[1]

    mat = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            xs, ys = extract(i, j, data)
            cc = get_correlation(xs, ys)
            mat[i, j] = cc

    return mat

def cache_data(data, fname='results/data_cache'):
    """ Save data for later processing steps
    """
    fdir = os.path.dirname(fname)
    if not os.path.isdir(fdir):
        os.makedirs(fdir)

    np.save(fname, data)

def extract_sig_entries(mat):
    """ Extract significant entries from correlation matrix (expects 3x3 matrix)
    """
    assert mat.shape == (3, 3), 'Matrix dimension mismatch ({})'.format(mat.shape)
    assert np.allclose(mat.transpose(), mat), 'Matrix not symmetric'

    ind = np.nonzero(np.tril(abs(mat), k=-1))
    res = mat[ind].tolist()
    res += [0] * (3-len(res))
    return np.array(res)

def extract_sub_matrix(mat, inds):
    """ Extract submatrix of `mat` by deleting `inds` rows/cols
    """
    for i in sorted(inds, reverse=True):
        mat = np.delete(mat, i, axis=0)
        mat = np.delete(mat, i, axis=1)
    return mat
