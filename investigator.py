"""
Investigate data in various ways
"""

import sys

import numpy as np
import matplotlib.pylab as plt

from utils import extract_sig_entries
from plotter import plot_histogram


def plot_correlation_hist(data):
    """ Plot histogram of all correlations
    """
    # gather data
    corrs = []
    for raw_res, enh_list in data:
        _, raw_mat, _ = raw_res

        if not raw_mat is None:
            raw_vec = extract_sig_entries(raw_mat)
            corrs.extend(raw_vec)

        for enh_res in enh_list:
            _, enh_mat, _ = enh_res

            if not enh_mat is None:
                enh_vec = extract_sig_entries(enh_mat)
                corrs.extend(enh_vec)

    # plot result
    fig = plt.figure()

    plot_histogram(corrs, plt.gca())
    plt.xlabel('simulated correlations')

    fig.savefig('images/all_sim_corrs.pdf')

def main(data):
    """ Analyse data
    """
    plot_correlation_hist(data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        sys.exit(-1)

    main(np.load(sys.argv[1])['data'])
