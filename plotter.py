"""
Visualization related functions
"""

import io, os

import numpy as np
import networkx as nx

import matplotlib.pylab as plt
import matplotlib.image as mpimg

from utils import get_correlation, extract


def save_figure(fname, **kwargs):
    """ Save current plot and check that directory exists
    """
    fdir = os.path.dirname(fname)
    if len(fdir) > 0 and not os.path.isdir(fdir):
        os.makedirs(fdir)

    plt.savefig(fname, **kwargs)

def plot_system_evolution(sol, ax):
    """ Plot solution of integration
    """
    for i, series in enumerate(sol):
        ax.plot(series, label=r'$S_%d$' % i)

    ax.set_ylim((0, ax.get_ylim()[1]))
    ax.set_xlabel('time')
    ax.set_ylabel('concentration')
    ax.legend(loc='best', ncol=2)

def plot_ss_scatter(steadies):
    """ Plot scatter plots of steady states
    """
    def do_scatter(i, j, ax):
        """ Draw single scatter plot
        """
        xs, ys = extract(i, j, steadies)
        ax.scatter(xs, ys)

        ax.set_xlabel(r'$S_%d$' % i)
        ax.set_ylabel(r'$S_%d$' % j)

        cc = get_correlation(xs, ys)
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
    save_figure('images/correlation_scatter.pdf', bbox_inches='tight')
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

    # add system interactions
    for i in range(dim):
        for j in range(dim):
            if J[i, j] != 0:
                graph.add_edge(j, i, label=round(J[i, j], 2))

    # mark external input
    for i, inp in enumerate(system.external_influence):
        if inp != 0:
            graph.add_node('ext_inp_%d' % i, style='invis')
            graph.add_edge(
                'ext_inp_%d' % i, i,
                color='"black:white:black"',
                label=round(inp, 2)) # "" are required

    # mark fluctuating nodes
    for i, fluc in enumerate(system.fluctuation_vector):
        if fluc != 0:
            graph.node[i]['shape'] = 'doublecircle'

    pydot_graph = nx.nx_pydot.to_pydot(graph)
    png_str = pydot_graph.create_png(prog=['dot', '-Gdpi=300'])
    img = mpimg.imread(io.BytesIO(png_str))

    ax.imshow(img, aspect='equal')
    ax.axis('off')
