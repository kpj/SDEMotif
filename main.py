"""
Infer metabolite correlation patterns from network motifs
"""

import os
import sys
import copy
import multiprocessing

import numpy as np
from tqdm import tqdm

import matplotlib.pylab as plt

from setup import load_systems, system_from_string
from solver import solve_system
from utils import compute_correlation_matrix, cache_data
from filters import filter_steady_state
from plotter import plot_system_evolution


def analyze_system(
    system, repetition_num=100,
    filter_trivial_ss=True, filter_mask=None,
    plot_hist=False, save_stdev=None,
    use_ode_sde_diff=True
):
    """ Generate steady states for given system.
        `filter_mask` is a list of nodes to be excluded from filtering.
        A filtered entry must have a None correlation matrix
    """
    if use_ode_sde_diff:
        ode_system = copy.copy(system)
        ode_system.fluctuation_vector = np.zeros(system.fluctuation_vector.shape)

    ss_data = []
    for _ in range(repetition_num):
        sde_sol = solve_system(system, tmax=100)
        if use_ode_sde_diff:
            ode_sol = solve_system(ode_system, tmax=100)

        if use_ode_sde_diff:
            sol = ode_sol - sde_sol
        else:
            sol = sde_sol
        sol_extract = sol.T[int(len(sol.T)*3/4):]

        if use_ode_sde_diff:
            ode_sol_extract = ode_sol.T[int(len(ode_sol.T)*3/4):]
        else:
            ode_sol_extract = sol_extract

        if not filter_trivial_ss or not filter_steady_state(ode_sol_extract, filter_mask):
            ss_data.append(sol_extract)
        else:
            return system, None, sol

    corr_mat = compute_correlation_matrix(np.array(ss_data), plot_hist, save_stdev)
    return system, corr_mat, sol

def cluster_data(data):
    """ Order data according to correlation matrices
    """
    return sorted(data, key=lambda e: np.sum(e[1]))

def main(fname):
    """ Main interface
    """
    if os.path.isfile(fname):
        systems = load_systems(fname)
        if systems.ndim == 0:
            systems = [np.asscalar(systems)]
        print('Integrating %d systems' % len(systems))

        core_num = int(multiprocessing.cpu_count() * 4/5)
        print('Using %d cores' % core_num)

        data = []
        with tqdm(total=len(systems)) as pbar:
            with multiprocessing.Pool(core_num) as p:
                for res in p.imap_unordered(analyze_system, systems, chunksize=10):
                    if not res[1] is None:
                        data.append(res)
                    pbar.update()
        print('Found result for %d systems' % len(data))

        data = cluster_data(data)
        cache_data(data)
    else:
        syst = system_from_string(fname)
        syst, mat, sol = analyze_system(syst, plot_hist=True)
        if mat is None:
            print('No sensible steady-state found')
        else:
            print(mat)
        plot_system_evolution(sol, plt.gca())
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <systems file | system spec>' % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1])
