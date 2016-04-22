"""
Visualize networks via results on axes
"""

import sys
import pickle

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt

from utils import extract_sig_entries
from plotter import save_figure
from network_matrix import plot_individuals


def transform(data):
    """ Convert input data into DataFrame
    """
    df = pd.DataFrame()
    for raw_res, enh_res_vec in data:
        df = df.append(
            [{
                'raw_res': raw_res,
                'enh_res': enh_res,
                'raw_mat': raw_res[1],
                'enh_mat': enh_res[1]
            } for enh_res in enh_res_vec],
            ignore_index=True)
    return df[~df.isnull().any(axis=1)]

def extract_entries(row):
    """ Extract needed amount of entries from each matrix
    """
    return pd.Series({
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'raw_vals': extract_sig_entries(row.raw_mat),
        'enh_vals': extract_sig_entries(row.enh_mat[:3,:3])
            if not row.enh_mat is None else None
    })

def bin_entries(row, threshold=0.2):
    """ Bin entries
    """
    def rm(vals):
        vals[abs(vals) < threshold] = 0
        return vals

    return pd.Series({
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'raw_vals': rm(row.raw_vals),
        'enh_vals': rm(row.enh_vals)
    })

def check_vanishing(row):
    """ Estimate number of vanishing correlations
    """
    mask = np.zeros(row.raw_vals.shape, dtype=bool)
    nv_inds = np.intersect1d(np.nonzero(row.raw_vals), np.nonzero(row.enh_vals))
    mask[nv_inds] = 1

    return pd.Series({
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'already_zero': np.sum(row.raw_vals == 0),
        'abs_corr_diff': np.mean(abs(row.raw_vals[mask] - row.enh_vals[mask]))
            if len(nv_inds) > 0 else np.nan
    })

def check_neg_pos_corrs(row):
    neg_pos = []
    pos_neg = []

    for raw, enh in zip(row.raw_vals, row.enh_vals):
        if raw < 0:
            neg_pos.append(abs(raw-enh))
        else:
            pos_neg.append(abs(raw-enh))

    return pd.Series({
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'mean_negpos_corr': np.mean(neg_pos),
        'mean_posneg_corr': np.mean(pos_neg)
    })

def plot_zeros_vs_corr_diff(df):
    """ Plot number of zero correlations in original network against mean of absolute correlation differences
    """
    df = df.apply(check_vanishing, axis=1)

    # generate overview
    plt.figure()

    sns.regplot(
        x='already_zero', y='abs_corr_diff', data=df,
        fit_reg=False,
        x_jitter=0.1)#, y_jitter=0.1)

    save_figure('images/network_space.pdf')

    # plot networks in detail
    extr = df[df.abs_corr_diff > 0.8]
    print(extr)

    mpl.style.use('default')
    plot_individuals(
        list(zip(extr.raw_res.tolist(), extr.enh_res.tolist())),
        'images/extreme.pdf')

def plot_neg_pos_corrs(df):
    """ Plot neg - pos corrs on x and pos - neg corrs on y axis
    """
    df = df.apply(check_neg_pos_corrs, axis=1)

    # generate overview
    plt.figure()
    sns.regplot(
        x='mean_negpos_corr', y='mean_posneg_corr', data=df,
        fit_reg=False)
    save_figure('images/network_space2.pdf')

    # plot networks in detail
    extr = df[(df.mean_negpos_corr > 0.6) & (df.mean_posneg_corr > 0.6)]
    print(extr)

    mpl.style.use('default')
    plot_individuals(
        list(zip(extr.raw_res.tolist(), extr.enh_res.tolist())),
        'images/extreme2.pdf')

def main(data):
    """ Analyse data
    """
    df = transform(data)
    df = df.apply(
        lambda row: bin_entries(extract_entries(row)),
        axis=1)

    plot_neg_pos_corrs(df)
    plot_zeros_vs_corr_diff(df)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        fname = sys.argv[1]
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
        main(inp['data'])
    else:
        print('Usage: {} <data file>'.format(sys.argv[0]))
        exit(1)
