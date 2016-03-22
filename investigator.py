"""
Investigate data in various ways
"""

import sys

import numpy as np

import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import trange

from utils import extract_sig_entries, compute_correlation_matrix
from plotter import plot_histogram
from setup import generate_basic_system
from solver import solve_system


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

def check_ergodicity(reps=500):
    """ Check whether simulated systems are ergodic
    """
    def get_matrices(syst, entry_num=100):
        """ Get correlation matrices for both cases
        """
        # multiple entries from single run
        single_run_matrices = []
        for _ in range(entry_num):
            sol = solve_system(syst)

            extract = sol.T[-entry_num:]
            single_run_mat = compute_correlation_matrix(np.array([extract]))

            single_run_matrices.append(single_run_mat)
        avg_single_mat = np.mean(single_run_matrices, axis=0)

        # one entry from multiple runs
        multiple_runs = []
        for _ in range(entry_num):
            sol = solve_system(syst)

            extract = sol.T[-1].T
            multiple_runs.append(extract)
        multiple_mat = compute_correlation_matrix(np.array([multiple_runs]))

        return avg_single_mat, multiple_mat

    syst = generate_basic_system()

    single_runs = []
    multiple_runs = []
    for _ in trange(reps):
        sm, rm = get_matrices(syst)

        single_runs.append(sm)
        multiple_runs.append(rm)
    single_runs = np.array(single_runs)
    multiple_runs = np.array(multiple_runs)

    # plot result
    dim = syst.jacobian.shape[1]

    plt.figure(figsize=(6, 14))
    gs = mpl.gridspec.GridSpec(int((dim**2-dim)/2), 1)

    axc = 0
    for i in range(dim):
        for j in range(dim):
            if i == j: break
            ax = plt.subplot(gs[axc])

            plot_histogram(
                single_runs[:,i,j], ax,
                alpha=0.5,
                label='Multiple entries from single run')
            plot_histogram(multiple_runs[:,i,j], ax,
                facecolor='mediumturquoise', alpha=0.5,
                label='One entry from multiple runs')

            ax.set_title('Nodes {}, {}'.format(i, j))
            ax.set_xlabel('correlation')
            ax.legend(loc='best')

            axc += 1

    plt.tight_layout()
    plt.savefig('images/ergodicity_check.pdf')

def main(data):
    """ Analyse data
    """
    if data is None:
        check_ergodicity()
    else:
        plot_correlation_hist(data)

if __name__ == '__main__':
    main(np.load(sys.argv[1])['data'] if len(sys.argv) == 2 else None)
