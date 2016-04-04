"""
Useful functions and more
"""

import os

import numpy as np
import scipy.stats as scis

import matplotlib as mpl
import matplotlib.pylab as plt

import plotter


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

def compute_correlation_matrix(data, plot_hist=False):
    """ Compute correlation matrix of given data points
    """
    dim = data.shape[2]

    mats = []
    for rep_slice in data:
        mat = np.empty((dim, dim))
        for i in range(dim):
            for j in range(dim):
                xs, ys = extract(i, j, rep_slice)
                cc = get_correlation(xs, ys)
                mat[i, j] = cc
        mats.append(mat)
    mats = np.array(mats)

    if plot_hist:
        plt.figure(figsize=(6, 14))
        gs = mpl.gridspec.GridSpec(int((dim**2-dim)/2), 1)

        axc = 0
        for i in range(dim):
            for j in range(dim):
                if i == j: break

                ax = plt.subplot(gs[axc])
                plotter.plot_histogram(mats[:,i,j], ax)
                ax.set_title('Nodes {}, {}'.format(i, j))
                ax.set_xlabel('correlation')

                axc += 1

        plt.tight_layout()
        plt.savefig('images/simulated_corrs_hist.pdf')
        plt.close()

    res_mat = np.mean(mats, axis=0)
    return res_mat

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
    if mat is None: return None
    assert np.allclose(mat.transpose(), mat), 'Matrix not symmetric'

    ind = np.nonzero(np.tril(abs(mat), k=-1))
    res = mat[ind].tolist()
    return np.array(res)

def extract_sub_matrix(mat, inds):
    """ Extract submatrix of `mat` by deleting `inds` rows/cols
    """
    for i in sorted(inds, reverse=True):
        mat = np.delete(mat, i, axis=0)
        mat = np.delete(mat, i, axis=1)
    return mat

def list_diff(l1, l2):
    """ Return set difference while maintaining order
    """
    return [x for x in l1 if x not in l2]
