"""
Investigate 3+1 node network with varied parameters
"""

import sys
import copy
import pickle
import itertools

import numpy as np
import matplotlib.pylab as plt

from tqdm import tqdm

from system import SDESystem
from main import analyze_system
from utils import extract_sig_entries


def generate_basic_system(v_in=5, k_m=1, k_23=2, D=1):
    """ Generate system according to paper
    """
    k_12 = k_13 = k_out = k_m

    jacobian = np.array([
        [-(k_12 + k_12),    0,      0],
        [k_12,              -k_23,  0],
        [k_13,              k_23,   -k_out]
    ])
    external_influence = np.array([v_in, 0, 0])
    fluctuation_vector = np.array([D, 0, 0])
    initial_state = np.array([1, 1, 1])

    system = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)
    return system

def add_node_to_system(syst):
    """ Add additional node to given system in all possible ways
    """
    tmp = copy.deepcopy(syst)

    # adjust vectors
    tmp.fluctuation_vector = np.append(tmp.fluctuation_vector, 0)
    tmp.external_influence = np.append(tmp.external_influence, 0)
    tmp.initial_state = np.append(tmp.initial_state, 1)

    # generate jacobians
    dim = tmp.jacobian.shape[0]

    horz_stacks = list(itertools.product([0, 1], repeat=dim))
    vert_stacks = list(itertools.product([0, 1], repeat=dim))

    systems = []
    for hs in horz_stacks:
        for vs in vert_stacks:
            cur = copy.deepcopy(tmp)
            vs = np.append(vs, -1) # force self-inhibition

            cur.jacobian = np.hstack(
                (cur.jacobian, np.array(hs).reshape(-1, 1)))
            cur.jacobian = np.vstack(
                (cur.jacobian, vs))

            systems.append(cur)
    return systems

def handle_systems(raw, enhanced):
    """ Simulate given systems
    """
    # generate control data
    raw_res = analyze_system(raw)

    row = []
    for enh in enhanced:
        enh_res = analyze_system(enh)
        row.append(enh_res)

    return [raw_res, row]

def generate_data(fname, paramter_shift=10):
    """ Generate and cache data of the form
        {
            'data': [
                [raw_res, [enh_res, ...]], # some parameter configuratoin
                ...
            ]
        }
    """
    param_range = np.linspace(0.1, 5, paramter_shift)

    rows = []
    for k_m in tqdm(param_range):
        for k_23 in tqdm(param_range, nested=True):
            syst = generate_basic_system(k_m=k_m, k_23=k_23)
            more = add_node_to_system(syst)

            res = handle_systems(syst, more)
            rows.append(res)

    # store matrix
    with open(fname, 'wb') as fd:
        pickle.dump({
            'data': rows,
        }, fd)

def preprocess_data(data, func):
    """ Extract data information
    """
    def handle_enh_entry(raw_res, enh_res):
        raw, raw_mat, raw_sol = raw_res
        enh, enh_mat, enh_sol = enh_res

        enh_mat = enh_mat[:-1,:-1] # disregard fourth node
        raw_vals = extract_sig_entries(raw_mat)
        enh_vals = extract_sig_entries(enh_mat)

        return func(raw_vals, enh_vals)

    plot_data = []
    for raw, enh_res in data:
        plot_data.append([handle_enh_entry(raw, enh) for enh in enh_res])

    return plot_data

def plot_result(inp, func):
    """ Plot generated matrix
    """
    # preprocess data
    data = preprocess_data(inp['data'], func)

    # create plot
    plt.xticks(np.arange(len(data[0]), dtype=np.int))
    plt.yticks(np.arange(len(data), dtype=np.int))

    plt.xlabel('networks')
    plt.ylabel('varied parameter')

    plt.imshow(
        data,
        interpolation='nearest')
    plt.colorbar()

    plt.savefig('images/matrix.pdf', bbox_inches='tight')
    plt.show()

def handle_plots(inp):
    """ Generate plots for varying data extraction functions
    """
    # define functions
    def get_sign_changes(raw_vals, enh_vals):
        """ Compute number of sign changes
        """
        return np.sum(np.sign(raw_vals) == np.sign(enh_vals))

    # do magic
    plot_result(inp, get_sign_changes)


def main():
    """ Create matrix for various data functions
    """
    fname = 'results/matrix_data.dat'

    if len(sys.argv) == 1:
        generate_data(fname)
    elif len(sys.argv) == 2:
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
        handle_plots(inp)
    else:
        print('Usage: %s [data file]' % sys.argv[0])
        sys.exit(-1)

if __name__ == '__main__':
    main()
