"""
Visualize networks via results on axes
"""

import sys
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

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

def plot_space(df):
    """ Plot result
    """
    sns.regplot(
        x='already_zero', y='abs_corr_diff', data=df,
        fit_reg=False,
        x_jitter=0.1)#, y_jitter=0.1)

    save_figure('images/network_space.pdf')

def main(data):
    """ Analyse data
    """
    df = transform(data)
    df = df.apply(
        lambda row: check_vanishing(bin_entries(extract_entries(row))),
        axis=1)

    plot_space(df)

    extr = df[df.abs_corr_diff>0.8]
    print(extr)

    mpl.style.use('default')
    plot_individuals(
        list(zip(extr.raw_res.tolist(), extr.enh_res.tolist())),
        'images/extreme.pdf')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fname = sys.argv[1]
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
        main(inp['data'])
    else:
        print('Usage: {} <data file>'.format(sys.argv[0]))
        exit(1)
