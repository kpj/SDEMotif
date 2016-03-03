"""
Infer metabolite correlation patterns from network motifs
"""

import sys
import multiprocessing

import numpy as np
from tqdm import tqdm

from setup import load_systems
from solver import solve_system, get_steady_state
from utils import compute_correlation_matrix, cache_data
from filters import filter_steady_state, filter_correlation_matrix


def analyze_system(system, repetition_num=100, filter_trivial_ss=True, filter_mask=None):
    """ Generate steady states for given system.
        `filter_mask` is a list of nodes to be excluded from filtering
    """
    ss_data = []
    for _ in range(repetition_num):
        sol = solve_system(system)

        ss = get_steady_state(sol)
        if not filter_trivial_ss or not filter_steady_state(ss, filter_mask):
            ss_data.append(ss)
        else:
            return system, None, None

    corr_mat = compute_correlation_matrix(np.array(ss_data))

    if filter_correlation_matrix(corr_mat, filter_mask):
        return system, None, None

    return system, corr_mat, sol

def cluster_data(data):
    """ Order data according to correlation matrices
    """
    return sorted(data, key=lambda e: np.sum(e[1]))

def main(fname):
    """ Main interface
    """
    systems = load_systems(fname)
    print('Integrating %d systems' % systems.size)

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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <systems file>' % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1])
