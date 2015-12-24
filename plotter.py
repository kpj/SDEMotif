"""
Visualization related functions
"""

import numpy as np
import scipy.stats as scis
import networkx as nx

import matplotlib.pylab as plt
from matplotlib import gridspec

from utils import get_nonconst_data


def plot_system_evolution(sol, ax):
    """ Plot solution of integration
    """
    for i, series in enumerate(sol):
        ax.plot(series, label=r'$S_%d$' % (i+1))

    ax.set_xlabel('time')
    ax.set_ylabel('concentration')
    ax.legend(loc='best')

def plot_ss_scatter(steadies):
    """ Plot scatter plots of steady states
    """
    def do_scatter(i, j, ax):
        """ Draw single scatter plot
        """
        xs, ys = get_nonconst_data(i, j, steadies)
        ax.scatter(xs, ys)

        ax.set_xlabel(r'$S_%d$' % (i+1))
        ax.set_ylabel(r'$S_%d$' % (j+1))

        cc, pval = scis.pearsonr(xs, ys)
        ax.set_title(r'Corr: $%.2f$' % cc)

    dim = steadies.shape[1]
    fig, axarr = plt.subplots(1, dim, figsize=(20, 5))

    axc = 0
    for i in range(dim):
        for j in range(dim):
            if i == j: break
            do_scatter(
                i, j,
                axarr[axc])
            axc += 1

    plt.suptitle('Correlation overview')

    plt.savefig('images/correlations.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_ss_heat(steadies, ax):
    """ Plot heatmap of steady state correlation coefficients
    """
    dim = steadies.shape[1]

    mat = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            xs, ys = get_nonconst_data(i, j, steadies)
            cc, pval = scis.pearsonr(xs, ys)
            mat[i, j] = cc

    # plot colors
    ax.set_xticks(np.arange(dim, dtype=np.int))
    ax.set_yticks(np.arange(dim, dtype=np.int))
    ax.imshow(
        mat,
        interpolation='nearest', cmap='bwr_r',
        vmin=-1, vmax=1)

    # add labels
    xms, yms = np.meshgrid(range(dim), range(dim))
    for i, j in zip(xms.flatten(), yms.flatten()):
        val = round(mat[i, j], 2)
        ax.text(i, j, val, va='center', ha='center')

def plot_system(system, ax):
    """ Plot network specified by Jacobian of system
    """
    J = system.jacobian
    graph = nx.DiGraph()

    dim = J.shape[0]
    graph.add_nodes_from(range(dim))

    edge_label_map = {}
    for i in range(dim):
        for j in range(dim):
            if J[i, j] != 0:
                graph.add_edge(j, i, weigth=J[i, j])
                edge_label_map[(j, i)] = round(J[i, j], 2)

    pos = nx.drawing.spring_layout(graph)
    nx.draw(
        graph, pos, ax=ax,
        with_labels=True)
    nx.draw_networkx_edge_labels(
        graph, pos, ax=ax,
        edge_labels=edge_label_map)

def plot_system_overview(data):
    """ Plot systems vs correlations
    """
    fig = plt.figure(figsize=(13, 4*len(data)))
    gs = gridspec.GridSpec(len(data), 3, width_ratios=[1, 1, 2])

    for i, (system, steadies, solution) in enumerate(data):
        plot_system(system, plt.subplot(gs[i, 0]))
        plot_ss_heat(steadies, plt.subplot(gs[i, 1]))
        plot_system_evolution(solution, plt.subplot(gs[i, 2]))

    plt.tight_layout()
    plt.savefig('images/overview.png', bbox_inches='tight', dpi=300)
    plt.close()
