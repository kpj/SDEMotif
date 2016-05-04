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

def check_disruption(row):

    sdiff = (np.sign(row.raw_vals) - np.sign(row.enh_vals)) * abs(row.raw_vals - row.enh_vals)

    return pd.Series({
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'sign_diff': np.mean(sdiff),
        'mean_difference': np.mean(abs(row.raw_vals - row.enh_vals))
    })

def plot_disruptiveness(df):
    """ Compare disruption of correlations (mean difference) to trivial disruption (1 mean(corr. of 4node network))
    """
    df = df.apply(check_disruption, axis=1)

    # generate overview
    plt.figure()
    sns.regplot(
        x='sign_diff', y='mean_difference', data=df,
        fit_reg=False)
    save_figure('images/network_space.pdf')

    # plot networks in detail
    extr = df[(df.sign_diff >= 0.3) & (df.mean_difference >= 0.4)]
    print(extr)

    if not extr.empty:
        mpl.style.use('default')
        plot_individuals(
            list(zip(extr.raw_res.tolist(), extr.enh_res.tolist())),
            'images/extreme.pdf')

def main(data):
    """ Analyse data
    """
    df = transform(data)
    df = df.apply(
        lambda row: bin_entries(extract_entries(row)),
        axis=1)

    plot_disruptiveness(df)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        fname = sys.argv[1]
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
        main(inp['data'])
    else:
        print('Usage: {} <data file>'.format(sys.argv[0]))
        exit(1)
