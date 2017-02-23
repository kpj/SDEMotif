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

    # generate final system
    system = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)
    #system.save('cache/embedded_system.pkl')
    return system

def plot_system(syst, ax):
    graph = nx.from_numpy_matrix(syst.jacobian.T, create_using=nx.DiGraph())
    nx.draw(
        graph, ax=ax, with_labels=True,
        node_color=['y' if n in range(3) else 'red' for n in graph.nodes()])

def plot_solution(sol, ax):
    for i, series in enumerate(sol):
        ax.plot(series, label=i, alpha=1 if i in range(3) else .3)
    ax.legend(loc='best')

def plot_correlation_hist(matrices, label, color, ax):
    for i, row in enumerate(matrices.T):
        for j, series in enumerate(row):
            if i == j: break
            sns.distplot(
                series, ax=ax, hist=False,
                label=label if i == 1 else None, color=color)
    ax.legend(loc='best')

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

def main():
    # compute data
    bas_syst = generate_basic_system()
    emb_syst = SDESystem.load('cache/embedded_system.pkl') #get_system(10)

    bas_mats, bas_extra = simulate(bas_syst, 30)
    emb_mats, emb_extra = simulate(emb_syst, 30)

    # generate plot
    plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 3, width_ratios=[2,4,2])

    plot_system(bas_extra[0], plt.subplot(gs[0,0]))
    plot_system(emb_extra[0], plt.subplot(gs[1,0]))

    plot_solution(bas_extra[1], plt.subplot(gs[0,1]))
    plot_solution(emb_extra[1], plt.subplot(gs[1,1]))

    plot_correlation_hist(bas_mats, 'only motif', 'blue', plt.subplot(gs[0,2]))
    plot_correlation_hist(emb_mats, 'embedded motif', 'red', plt.subplot(gs[1,2]))

    plt.tight_layout()
    plt.savefig('images/embedded_motif.pdf')


if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    main()
