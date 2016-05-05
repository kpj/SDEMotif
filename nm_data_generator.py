"""
Generate data to be used with e.g. network_space.py
"""

import sys
import copy
import pickle
import itertools

import numpy as np
from tqdm import tqdm

from setup import generate_basic_system
from main import analyze_system


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
    raw_res_diff = analyze_system(raw, filter_mask=[3])
    raw_res = analyze_system(raw, filter_mask=[3], use_ode_sde_diff=False)
    if raw_res[1] is None:
        return None

    # generate data from altered motifs
    row = []
    for enh in enhanced:
        enh_res_diff = analyze_system(enh, filter_mask=[3])
        enh_res = analyze_system(enh, filter_mask=[3], use_ode_sde_diff=False)
        row.append((enh_res, enh_res_diff))

    return [(raw_res, raw_res_diff), row]

def generate_data(fname, paramter_shift=10):
    """ Generate and cache data of the form
        {
            'data': [
                [raw_res, [enh_res, ...]], # some parameter configuration
                ...
            ] # rows in output plot
        }
    """
    param_range = np.linspace(0.1, 5, paramter_shift)

    # iterate over parameter configurations and simulate system accordingly
    rows = []
    for k_m in tqdm(param_range):
        for k_23 in tqdm(param_range):
            syst = generate_basic_system(k_m=k_m, k_23=k_23)
            more = add_node_to_system(syst)

            res = handle_systems(syst, more)
            if not res is None:
                rows.append(res)

    # store matrix
    with open(fname, 'wb') as fd:
        pickle.dump({
            'data': rows
        }, fd)


def main():
    """
    Main interface
    """
    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        exit(-1)

    generate_data(sys.argv[1])

if __name__ == '__main__':
    main()
