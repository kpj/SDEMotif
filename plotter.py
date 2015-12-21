"""
Visualization related functions
"""

import numpy as np
import scipy.stats as scis

import matplotlib.pylab as plt
import networkx as nx


def plot_system_evolution(sol):
    """ Plot solution of integration
    """
    plt.figure(figsize=(10, 5))

    for i, series in enumerate(sol):
        plt.plot(series, label=r'$S_%d$' % (i+1))

    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(loc='best')

    plt.savefig('images/evolution.png', bbox_inches='tight', dpi=300)
    #plt.show()

def plot_ss_scatter(steadies):
    """ Plot scatter plots of steady states
    """
    def do_scatter(i, j, ax):
        """ Draw single scatter plot
        """
        xs, ys = steadies[:,i], steadies[:,j]

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
    #plt.show()

def plot_ss_heat(steadies, ax):
    """ Plot heatmap of steady state correlation coefficients
    """
    dim = steadies.shape[1]

    mat = np.empty((dim, dim))
    for i in range(dim):
        for j in range(dim):
            xs, ys = steadies[:,i], steadies[:,j]
            cc, pval = scis.pearsonr(xs, ys)
            mat[i, j] = cc

    ax.imshow(
        mat,
        interpolation='nearest', cmap='bwr_r',
        vmin=-1, vmax=1)

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
                graph.add_edge(i, j, weigth=J[i, j])
                edge_label_map[(i, j)] = J[i, j]

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
    fac = 5
    fig, axarr = plt.subplots(len(data), 2, figsize=(fac, fac*len(data)))

    for i, (system, steadies) in enumerate(data):
        plot_system(system, axarr[i][0])
        plot_ss_heat(steadies, axarr[i][1])

    plt.tight_layout()
    plt.savefig('images/overview.png', bbox_inches='tight', dpi=300)
    #plt.show()
