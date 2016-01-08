"""
Data processing facilities
"""

import collections

import numpy as np

import matplotlib.pylab as plt
from matplotlib import gridspec

from plotter import save_figure, plot_system, plot_corr_mat, plot_system_evolution


def plot_system_overview(data):
    """ Plot systems vs correlations
    """
    fig = plt.figure(figsize=(13, 4*len(data)))
    gs = gridspec.GridSpec(len(data), 3, width_ratios=[1, 1, 2])

    for i, (system, corr_mat, solution) in enumerate(data):
        plot_system(system, plt.subplot(gs[i, 0]))
        plot_corr_mat(corr_mat, plt.subplot(gs[i, 1]))
        plot_system_evolution(solution, plt.subplot(gs[i, 2]))

    plt.tight_layout()
    save_figure('images/overview.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def network_density(data):
    """ Plot network edge density vs correlation quotient
    """
    points = collections.defaultdict(list)
    for syst, mat, _ in data:
        max_edge_num = syst.jacobian.shape[0] * (syst.jacobian.shape[0]+1)
        dens = np.count_nonzero(syst.jacobian) / max_edge_num
        quot = mat[0, 2] / mat[1, 2] if mat[1, 2] != 0 else 0
        points[dens].append(quot)

    densities = []
    quotients = []
    errbars = []
    for dens, quots in points.items():
        densities.append(dens)
        quotients.append(np.mean(quots))
        errbars.append(np.std(quots))

    plt.errorbar(
        densities, quotients, yerr=errbars,
        fmt='o', clip_on=False)

    plt.xlabel('motif edge density')
    plt.ylabel('correlation quotient')

    plt.tight_layout()
    save_figure('images/edens_quot.pdf', bbox_inches='tight')
    plt.close()

def main():
    """ Main interface
    """
    data = np.load('results/data_cache.npy')

    plot_system_overview(data)
    network_density(data)


if __name__ == '__main__':
    main()
