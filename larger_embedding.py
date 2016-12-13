"""
Embed motif in larger network
"""

import copy

import numpy as np
import networkx as nx
import scipy.stats as scis
import matplotlib.pyplot as plt

from tqdm import trange

from system import SDESystem
from solver import solve_system
from filters import filter_steady_state


def get_system(N, v_in=5, D=1):
    assert N >= 3, 'Cannot add FFL'

    graph = nx.barabasi_albert_graph(N, 1)
    graph.add_edges_from([(0,1),(1,2),(0,2)]) # add FFL
    jacobian = np.asarray(nx.to_numpy_matrix(graph))

    np.fill_diagonal(jacobian, -1)

    external_influence = np.ones(N) * v_in/N
    fluctuation_vector = np.ones(N) * D/N
    initial_state = np.ones(N)

    # drive FFL
    external_influence[0] = v_in
    fluctuation_vector[0] = D

    system = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)
    return system

def simulate_system(sde_system, reps=100):
    ode_system = copy.copy(sde_system)
    ode_system.fluctuation_vector = np.zeros(sde_system.fluctuation_vector.shape)

    corr_mats = []
    for _ in trange(reps):
        sde_sol = solve_system(sde_system)
        ode_sol = solve_system(ode_system)

        sol = ode_sol - sde_sol
        sol_extract = sol.T[int(len(sol.T)*3/4):] # extract steady-state

        plt.plot(sol_extract)
        plt.show()
        exit()

        # if filter_steady_state(ode_sol.T[int(len(ode_sol.T)*3/4):]):
            # continue

        # compute correlations
        dim = sol_extract.shape[1]
        mat = np.empty((dim,dim))
        for i in range(dim):
            for j in range(dim):
                xs, ys = sol_extract[:,i], sol_extract[:,j]
                cc, pval = scis.pearsonr(xs, ys)
                mat[i,j] = cc

        corr_mats.append(mat)
    return np.asarray(corr_mats)

def main():
    syst = get_system(10)
    corr_mats = simulate_system(syst)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
