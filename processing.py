"""
Data processing facilities
"""

import sys
import collections

import numpy as np
import networkx as nx

import matplotlib.pylab as plt
from matplotlib import gridspec

from plotter import save_figure, plot_system, plot_corr_mat, plot_system_evolution
from utils import get_correlation


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

def node_degree(data, bin_num_x=100, bin_num_y=100):
    """ Compare node degree and correlation
    """
    # get data
    ndegs = []
    avg_corrs = []
    node_num = -1
    for syst, mat, _ in data:
        graph = nx.DiGraph(syst.jacobian)
        for i in graph.nodes():
            ndegs.append(graph.degree(i))
            avg_corrs.append(
                np.mean([abs(mat[i, j]) for j in graph.nodes() if i != j]))
        node_num = graph.number_of_nodes()
    assert node_num >= 0, 'Invalid data found'

    # plot data
    heatmap, xedges, yedges = np.histogram2d(
        avg_corrs, ndegs,
        bins=(bin_num_x, bin_num_y))
    extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
    heatmap = heatmap[::-1]
    plt.imshow(
        heatmap,
        extent=extent, interpolation='nearest',
        aspect=abs((extent[1]-extent[0])/(extent[3]-extent[2])))
    plt.colorbar()

    cc = get_correlation(ndegs, avg_corrs)
    plt.title(r'Corr: $%.2f$' % cc)

    plt.xlabel('node degree')
    plt.ylabel('average absolute correlation to other nodes')

    plt.tight_layout()
    save_figure('images/ndegree_corr.pdf', bbox_inches='tight')
    plt.close()

def main(fname, data_step=1):
    """ Main interface
    """
    data = np.load(fname)[::data_step]

    plot_system_overview(data)
    network_density(data)
    node_degree(data)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1])
