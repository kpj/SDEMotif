"""
 Create nice looking publication figures
"""

import numpy as np
import networkx as nx

import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import tqdm

from main import analyze_system
from setup import generate_basic_system
from nm_data_generator import add_node_to_system


def visualize_node_influence():
    """ Compare examples where fourth node perturbs system and where it doesn't
    """
    def simulate(syst, reps=1000):
        matrices = []
        with tqdm(total=reps) as pbar:
            while reps >= 0:
                _, mat, _ = analyze_system(syst, repetition_num=1)
                if mat is None:
                    continue
                pbar.update()
                reps -= 1

                if mat.shape == (4, 4):
                    mat = mat[:-1, :-1]
                assert mat.shape == (3, 3)

                matrices.append(mat)
        return np.asarray(matrices)

    def plot_correlation_hist(matrices, ax):
        for i, row in enumerate(matrices.T):
            for j, series in enumerate(row):
                if i == j: break

                sns.distplot(series, ax=ax, label=r'$c_{{{},{}}}$'.format(i,j))

        ax.set_xlabel('correlation')
        ax.set_ylabel('count')
        ax.set_xlim((-1,1))
        ax.legend(loc='best')

    def plot_system(syst, ax):
        graph = nx.from_numpy_matrix(syst.jacobian.T, create_using=nx.DiGraph())
        nx.draw(
            graph, ax=ax,
            with_labels=True)
        ax.axis('off')
        ax.set_xticks([], [])
        ax.set_yticks([], [])

    # generate systems
    basic_system = generate_basic_system()
    more_systs = add_node_to_system(basic_system)

    similar_system = more_systs[42]
    different_system = more_systs[22] # 52

    systems = {
        'basic': basic_system,
        'similar': similar_system,
        'different_system': different_system
    }

    # simulations
    matrices = {}
    for name, syst in systems.items():
        matrices[name] = (syst, simulate(syst))

    # plot result
    for name, (syst, mats) in matrices.items():
        plt.figure()
        plot_correlation_hist(mats, plt.gca())
        plot_system(syst, plt.axes([.3,.5,.3,.3]))
        plt.savefig(f'images/node_influence_{name}.pdf')

def main():
    visualize_node_influence()


if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    main()
