"""
Embed motif in larger network
"""

import copy
import itertools

import numpy as np
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm

from system import SDESystem
from main import analyze_system
from setup import generate_basic_system


def get_system(N, v_in=5, D=1):
    assert N >= 3, 'Cannot add FFL'
    graph = nx.DiGraph(nx.scale_free_graph(N))

    # add FFL
    graph.remove_edges_from(itertools.product(range(3), repeat=2))
    graph.add_edges_from([(0,1),(1,2),(0,2)])

    jacobian = np.asarray(nx.to_numpy_matrix(graph)).T

    np.fill_diagonal(jacobian, -1)

    external_influence = np.random.randint(0, 2, size=N) * v_in
    fluctuation_vector = np.random.randint(0, 2, size=N) * D
    initial_state = np.ones(N)

    # drive FFL
    external_influence[0] = v_in
    fluctuation_vector[0] = D
    external_influence[[1,2]] = 0
    fluctuation_vector[[1,2]] = 0

    jacobian[0,0] = -2
    jacobian[1,1] = -2
    jacobian[2,1] = 2

    # generate final system
    system = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)
    #system.save('cache/embedded_system.pkl')
    return system

def plot_system(syst, ax):
    graph = nx.from_numpy_matrix(syst.jacobian.T, create_using=nx.DiGraph())
    pos = nx.circular_layout(graph)

    nx.draw(
        graph, pos,
        ax=ax, with_labels=True,
        node_color=['y' if n in range(3) else 'red' for n in graph.nodes()])
    nx.draw_networkx_edge_labels(
        graph, pos,
        edge_labels={(u,v): d['weight'] for u,v,d in graph.edges_iter(data=True)})

def plot_solution(sol, ax):
    for i, series in enumerate(sol):
        ax.plot(series, label=i, alpha=1 if i in range(3) else .3)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:3], labels[:3],
        loc='upper left', prop={'size':8})

def plot_correlation_hist(matrices, ax, shade=False, color_dict={}):
    for i, row in enumerate(matrices.T):
        for j, series in enumerate(row):
            if i == j: break
            g = sns.distplot(
                series, ax=ax, hist=False,
                kde_kws={'shade':shade, 'linewidth':0 if shade else None},
                color=color_dict[(i,j)] if (i,j) in color_dict else None,
                label=rf'$c_{{{i},{j}}}$')

            if (i,j) not in color_dict:
                cur_col = g.get_lines()[-1].get_color()
                color_dict[(i,j)] = cur_col

    ax.set_xlim((-1,1))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc='upper left')

def simulate(syst, reps=1000):
    matrices = []
    with tqdm(total=reps) as pbar:
        while reps >= 0:
            _, mat, sol = analyze_system(syst, filter_trivial_ss=False, repetition_num=1)
            if mat is None:
                continue
            pbar.update()
            reps -= 1

            x,y = mat.shape
            if mat.shape != (3,3):
                mat = mat[:-(x-3), :-(y-3)]
            assert mat.shape == (3, 3), mat.shape

            matrices.append(mat)
    return np.asarray(matrices), (syst, sol)

def add_fourth_node(bas_syst, emb_syst):
    """ Add fourth node according to embedding
    """
    bas_syst = copy.copy(bas_syst)

    J = bas_syst.jacobian
    eJ = emb_syst.jacobian

    driver_infl = (emb_syst.fluctuation_vector!=0)[3:] | True
    fnode_out = (driver_infl * eJ[:3,3:]).sum(axis=1)
    fnode_in = (driver_infl * eJ[3:,:3].T).sum(axis=1)

    J = np.vstack((J, fnode_in))
    J = np.hstack((J, np.r_[fnode_out, -1].reshape(-1,1)))

    bas_syst.jacobian = J
    bas_syst.fluctuation_vector = np.r_[bas_syst.fluctuation_vector, 1]
    bas_syst.external_influence = np.r_[bas_syst.external_influence, 5]
    bas_syst.initial_state = np.r_[bas_syst.initial_state, 1]

    return bas_syst

def main():
    # compute data
    bas_syst = generate_basic_system()
    fno_syst = add_fourth_node(bas_syst, emb_syst)
    assert (emb_syst.jacobian[:3,:3] == fno_syst.jacobian[:3,:3]).all()

    bas_mats, bas_extra = simulate(bas_syst, 100)
    fno_mats, fno_extra = simulate(fno_syst, 100)
    emb_mats, emb_extra = simulate(emb_syst, 100)

    # generate plot
    plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2,4,2])

    plot_system(fno_extra[0], plt.subplot(gs[0,0]))
    plot_system(emb_extra[0], plt.subplot(gs[1,0]))

    plot_solution(fno_extra[1], plt.subplot(gs[0,1]))
    plot_solution(emb_extra[1], plt.subplot(gs[1,1]))

    plot_correlation_hist(fno_mats, plt.subplot(gs[0,2]))
    plot_correlation_hist(emb_mats, plt.subplot(gs[1,2]))

    plot_correlation_hist(bas_mats, plt.subplot(gs[0,2]), shade=True)
    plot_correlation_hist(bas_mats, plt.subplot(gs[1,2]), shade=True)

    plt.tight_layout()
    plt.savefig('images/embedded_motif.pdf')


if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    main()
