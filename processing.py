"""
Data processing facilities
"""

import sys
import random
import itertools
import collections

import numpy as np
import networkx as nx

import matplotlib.pylab as plt
from matplotlib import gridspec

from plotter import save_figure, plot_system, plot_corr_mat, plot_system_evolution
from utils import get_correlation


def plot_system_overview(data, sample_size=20):
    """ Plot systems vs correlations
    """
    # extract sample
    dsample = [data[i]
        for i in sorted(random.sample(range(len(data)), min(len(data), sample_size)))]

    # plot sample
    fig = plt.figure(figsize=(13, 4*len(dsample)))
    gs = gridspec.GridSpec(len(dsample), 3, width_ratios=[1, 1, 2])

    for i, (system, corr_mat, solution) in enumerate(dsample):
        plot_system(system, plt.subplot(gs[i, 0]))
        plot_corr_mat(corr_mat, plt.subplot(gs[i, 1]))
        plot_system_evolution(solution, plt.subplot(gs[i, 2]))

    plt.tight_layout()
    save_figure('images/overview.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def network_density(data):
    """ Plot network edge density vs correlation quotient
    """
    def gen(it):
        """ Compute all possible heterogeneous pairs of `it`
        """
        return filter(lambda e: e[0] < e[1], itertools.product(it, repeat=2))

    points = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for syst, mat, _ in data:
        max_edge_num = syst.jacobian.shape[0] * (syst.jacobian.shape[0]+1)
        dens = np.count_nonzero(syst.jacobian) / max_edge_num

        dim = syst.jacobian.shape[0]
        indices = gen(range(dim))
        quot_pairs = gen(indices)

        for pair in quot_pairs:
            p1, p2 = pair
            quot = mat[p1] / mat[p2] if mat[p2] != 0 else 0
            points[pair][dens].append(quot)

    # plot figure
    fig = plt.figure(figsize=(6, 4*len(points)))
    gs = gridspec.GridSpec(len(points), 1)

    def plot(ax, data, title):
        """ Plot given data
        """
        densities = []
        quotients = []
        errbars = []
        for dens, quots in data.items():
            densities.append(dens)
            quotients.append(np.mean(quots))
            errbars.append(np.std(quots))

        ax.errorbar(
            densities, quotients, yerr=errbars,
            fmt='o', clip_on=False)

        ax.set_title(title)
        ax.set_xlabel('motif edge density')
        ax.set_ylabel('correlation quotient')

    for i, (spec, dat) in enumerate(points.items()):
        (x1, y1), (x2, y2) = spec
        plot(plt.subplot(gs[i]), dat,
            r'Quotient: $C_{%d%d} / C_{%d%d}$' % (x1, y1, x2, y2))

    plt.tight_layout()
    save_figure('images/edens_quot.pdf', bbox_inches='tight')
    plt.close()

def network_density_avg(data):
    """ Plot network edge density vs absolute, average correlation quotient
    """
    points = collections.defaultdict(list)
    for syst, mat, _ in data:
        max_edge_num = syst.jacobian.shape[0] * (syst.jacobian.shape[0]+1)
        dens = np.count_nonzero(syst.jacobian) / max_edge_num

        ind = np.nonzero(np.tril(abs(mat), k=-1))
        avg = np.mean(mat[ind])

        points[dens].append(avg)

    # plot figure
    densities = []
    averages = []
    errbars = []
    for dens, avgs in points.items():
        densities.append(dens)
        averages.append(np.mean(avgs))
        errbars.append(np.std(avgs))

    plt.errorbar(
        densities, averages, yerr=errbars,
        fmt='o', clip_on=False)

    plt.title('')
    plt.xlabel('motif edge density')
    plt.ylabel('(absolute) average node correlation')

    plt.tight_layout()
    save_figure('images/edens_avg.pdf', bbox_inches='tight')
    plt.close()

def network_density_clustering(data):
    """ Plot network edge density vs clustering coefficient
    """
    points = collections.defaultdict(list)
    for syst, mat, _ in data:
        max_edge_num = syst.jacobian.shape[0] * (syst.jacobian.shape[0]+1)
        dens = np.count_nonzero(syst.jacobian) / max_edge_num

        graph = nx.from_numpy_matrix(syst.jacobian)
        clus = nx.average_clustering(graph)

        points[dens].append(clus)

    # plot figure
    densities = []
    averages = []
    errbars = []
    for dens, avgs in points.items():
        densities.append(dens)
        averages.append(np.mean(avgs))
        errbars.append(np.std(avgs))

    plt.errorbar(
        densities, averages, yerr=errbars,
        fmt='o', clip_on=False)

    plt.title('')
    plt.xlabel('motif edge density')
    plt.ylabel('clustering coefficient')

    plt.tight_layout()
    save_figure('images/edens_clus.pdf', bbox_inches='tight')
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
            ncorrs = [abs(mat[i, j]) for j in graph.neighbors(i) if i != j]
            avg_corrs.append(
                np.mean(ncorrs) if len(ncorrs) > 0 else 0)
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
    network_density_avg(data)
    network_density_clustering(data)
    node_degree(data)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1])
