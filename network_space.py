"""
Visualize networks via results on axes
"""

import sys
import pickle

import numpy as np
import pandas as pd
import seaborn as sns

from utils import extract_sig_entries
from plotter import save_figure


def transform(data):
    """ Convert input data into DataFrame
    """
    df = pd.DataFrame(columns=['raw_mat', 'enh_mat'])
    for raw_res, enh_res_vec in data:
        df = df.append(
            [{
                'raw_mat': raw_res[1],
                'enh_mat': e[1]
            } for e in enh_res_vec],
            ignore_index=True)
    return df[~df.isnull().any(axis=1)]

def extract_entries(row):
    """ Extract needed amount of entries from each matrix
    """
    return pd.Series({
        'raw_vals': extract_sig_entries(row.raw_mat),
        'enh_vals': extract_sig_entries(row.enh_mat[:3,:3]
            if not row.enh_mat is None else None)
    })

def bin_entries(row, threshold=0.2):
    """ Bin entries
    """
    def rm(vals):
        vals[abs(vals) < threshold] = 0
        return vals

    return pd.Series({
        'raw_vals': rm(row.raw_vals),
        'enh_vals': rm(row.enh_vals)
    })

def check_vanishing(row):
    """ Estimate number of vanishing correlations
    """
    assert len(np.nonzero(row.raw_vals)) == len(np.nonzero(row.enh_vals))

    return pd.Series({
        'already_zero': np.sum(row.raw_vals == 0),
        'vanished': np.sum((row.raw_vals != 0) & (row.enh_vals == 0))
    })

def plot_space(df):
    """ Plot result
    """
    sns.regplot(
        x='already_zero', y='vanished', data=df,
        fit_reg=False,
        x_jitter=0.4, y_jitter=0.4)

    save_figure('images/network_space.pdf')
    save_figure('images/network_space.png')

def main(data):
    """ Analyse data
    """
    df = transform(data)
    df = df.apply(
        lambda row: check_vanishing(bin_entries(extract_entries(row))),
        axis=1)

    plot_space(df)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fname = sys.argv[1]
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
        main(inp['data'])
    else:
        print('Usage: {} <data file>'.format(sys.argv[0]))
        exit(1)
