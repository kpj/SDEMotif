"""
Visualization related functions
"""

import io

import numpy as np
import scipy.stats as scis
import networkx as nx

import matplotlib.pylab as plt
from matplotlib import gridspec
import matplotlib.image as mpimg

from utils import get_nonconst_data


def plot_system_evolution(sol, ax):
    """ Plot solution of integration
    """
    for i, series in enumerate(sol):
        ax.plot(series, label=r'$S_%d$' % i)

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

        ax.set_xlabel(r'$S_%d$' % i)
        ax.set_ylabel(r'$S_%d$' % j)

        cc, pval = scis.pearsonr(xs, ys)
        ax.set_title(r'Corr: $%.2f$' % cc)

    dim = steadies.shape[1]
    fig, axarr = plt.subplots(1, int((dim**2-dim)/2), figsize=(20, 5))

    axc = 0
    for i in range(dim):
        for j in range(dim):
            if i == j: break
            do_scatter(
                i, j,
                axarr[axc])
            axc += 1

    plt.suptitle('Correlation overview')

    plt.tight_layout()
    plt.savefig('images/correlation_scatter.pdf', bbox_inches='tight')
    plt.close()

def plot_corr_mat(corr_mat, ax):
    """ Plot heatmap of steady state correlation coefficients
    """
    dim = corr_mat.shape[0]

    # plot colors
    ax.set_xticks(np.arange(dim, dtype=np.int))
    ax.set_yticks(np.arange(dim, dtype=np.int))
    ax.imshow(
        corr_mat,
        interpolation='nearest', cmap='bwr_r',
        vmin=-1, vmax=1)

    # add labels
    xms, yms = np.meshgrid(range(dim), range(dim))
    for i, j in zip(xms.flatten(), yms.flatten()):
        val = round(corr_mat[i, j], 2)
        ax.text(i, j, val, va='center', ha='center')

def plot_system(system, ax):
    """ Plot network specified by Jacobian of system
    """
    J = system.jacobian
    graph = nx.DiGraph()

    dim = J.shape[0]
    graph.add_nodes_from(range(dim))

    for i in range(dim):
        for j in range(dim):
            if J[i, j] != 0:
                graph.add_edge(j, i, label=round(J[i, j], 2))

    pydot_graph = nx.to_pydot(graph)
    png_str = pydot_graph.create_png(prog=['dot', '-Gdpi=300'])
    img = mpimg.imread(io.BytesIO(png_str))

    ax.imshow(img, aspect='equal')
    ax.axis('off')

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
    plt.savefig('images/overview.pdf', bbox_inches='tight')
    plt.close()
