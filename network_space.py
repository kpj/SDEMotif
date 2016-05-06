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
from plotter import *


def transform(data):
    """ Convert input data into DataFrame
    """
    tmp = []
    count = 0
    for (raw_res, raw_res_diff), enh_res_vec in data:
        for (enh_res, enh_res_diff) in enh_res_vec:
            tmp.append({
                'type': 'ODE',
                'id': count,
                'raw_res': raw_res,
                'enh_res': enh_res,
                'raw_mat': raw_res[1],
                'enh_mat': enh_res[1]
            })
            tmp.append({
                'type': 'ODE-SDE',
                'id': count,
                'raw_res': raw_res_diff,
                'enh_res': enh_res_diff,
                'raw_mat': raw_res_diff[1],
                'enh_mat': enh_res_diff[1]
            })

            count += 1

    df = pd.DataFrame.from_dict(tmp)
    return df[~df.isnull().any(axis=1)]

def extract_entries(row):
    """ Extract needed amount of entries from each matrix
    """
    return pd.Series({
        'type': row.type,
        'id': row.id,
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'raw_vals': extract_sig_entries(row.raw_mat),
        'enh_vals': extract_sig_entries(row.enh_mat[:3,:3])
    })

def bin_entries(row, threshold=0.2):
    """ Bin entries
    """
    def rm(vals):
        vals[abs(vals) < threshold] = 0
        return vals

    return pd.Series({
        'type': row.type,
        'id': row.id,
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'raw_vals': rm(row.raw_vals),
        'enh_vals': rm(row.enh_vals)
    })

def check_disruption(row):
    sdiff = (np.sign(row.raw_vals) - np.sign(row.enh_vals)) * abs(row.raw_vals - row.enh_vals)

    return pd.Series({
        'type': row.type,
        'id': row.id,
        'raw_res': row.raw_res,
        'enh_res': row.enh_res,
        'sign_diff': abs(np.mean(sdiff)),
        'mean_difference': np.mean(abs(row.raw_vals - row.enh_vals))
    })

def plot_extract(df, bak_df, fname):
    """ Plot a selection of individual results
    """
    # plot selected networks
    fig = plt.figure(figsize=(50, 4*df.shape[0]))
    gs = mpl.gridspec.GridSpec(
        df.shape[0], 7,
        width_ratios=[1, 2, 1, 2, 4, 2, 4])

    for i, (_, row) in enumerate(df.iterrows()):
        raw, enh = row.raw_res, row.enh_res

        plot_system(raw[0], plt.subplot(gs[i, 0]))
        plot_corr_mat(raw[1], plt.subplot(gs[i, 1]))
        plot_system(enh[0], plt.subplot(gs[i, 2]))
        plot_corr_mat(enh[1], plt.subplot(gs[i, 3]))
        plot_system_evolution(enh[2], plt.subplot(gs[i, 4]))

        # compare to other solution
        other = bak_df[(bak_df.id == row.id) & (bak_df.type != row.type)]
        assert other.shape[0] == 1
        other = other.iloc[0]
        raw, enh = other.raw_res, other.enh_res

        plot_corr_mat(enh[1], plt.subplot(gs[i, 5]))
        plot_system_evolution(enh[2], plt.subplot(gs[i, 6]))

    plt.tight_layout()
    save_figure('{}_zoom.pdf'.format(fname.replace('.pdf', '')), bbox_inches='tight', dpi=300)
    plt.close()

def plot_disruptiveness(df):
    """ Compare disruption of correlations (mean difference) to trivial disruption (1 mean(corr. of 4node network))
    """
    df = df.apply(check_disruption, axis=1)

    rm_id = df[df.sign_diff == 0].id
    df = df[~df.id.isin(rm_id)]

    # generate overview
    plt.figure()
    sns.lmplot(
        x='sign_diff', y='mean_difference', data=df,
        col='type', fit_reg=False)
    save_figure('images/network_space.pdf')

    # plot networks in detail
    sub_df = df[(df.sign_diff >= 0) & (df.mean_difference >= .3)]
    #print(sub_df)

    if not sub_df.empty:
        mpl.style.use('default')
        plot_extract(sub_df, df, 'images/extreme.pdf')

def main(data):
    """ Analyse data
    """
    df = transform(data)
    print('Transformed data...')
    #df = df.apply(
    #    lambda row: bin_entries(extract_entries(row)),
    #    axis=1)
    df = df.apply(extract_entries, axis=1)
    print('Extracted entries...')

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
