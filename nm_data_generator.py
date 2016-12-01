"""
Generate data to be used with e.g. network_space.py
"""

import os
import sys
import copy
import pickle
import itertools

from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import numpy.random as npr
from tqdm import tqdm, trange

from setup import generate_basic_system, generate_two_node_system, generate_motifs
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
    raw_res_diff = analyze_system(raw, filter_mask=[3], use_ode_sde_diff=True, save_stdev='results/corr_stdev')
    raw_res = analyze_system(raw, filter_mask=[3], use_ode_sde_diff=False)
    if raw_res[1] is None or raw_res_diff[1] is None:
        return None

    # generate data from altered motifs
    row = []
    for enh in enhanced:
        enh_res_diff = analyze_system(
            enh, filter_mask=[3],
            use_ode_sde_diff=True)
        enh_res = analyze_system(
            enh, filter_mask=[3],
            use_ode_sde_diff=False)
        row.append((enh_res, enh_res_diff))

    return [(raw_res, raw_res_diff), row]

def generate_system_data(motifs_three):
    """ Generate data for a given system
    """
    res = []
    for motif in motifs_three:
        getter = lambda k_m, k_23: motif
        cur = generate_data(None, gen_func=getter)
        res.append(cur)
    return (getter(1,1), res)

def generate_motif_data(prefix):
    """ Generate data for all motifs
    """
    motifs = generate_motifs()
    with tqdm(total=len(motifs)) as pbar:
        resolution = int(cpu_count() * 3/4)
        with ThreadPool(resolution) as p:
            for motif, rows_three in p.map(generate_system_data, motifs):
                fname = '{}_{}'.format(prefix, pbar.n)
                with open(fname, 'wb') as fd:
                    pickle.dump({
                        'data': rows_three,
                        'motif': motif
                    }, fd)
                pbar.update()


def generate_data(fname, gen_func=generate_basic_system, paramter_shift=10):
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
    configurations = []
    for k_m in param_range:
        for k_23 in param_range:
            syst = gen_func(k_m=k_m, k_23=k_23)
            more = add_node_to_system(syst)
            configurations.append((syst, more))

        # only one parameter to vary in case of two nodes
        if gen_func == generate_two_node_system:
            break

    rows = []
    with tqdm(total=len(configurations)) as pbar:
        resolution = int(cpu_count() * 3/4)
        with Pool(resolution) as p:
            for res in p.starmap(handle_systems, configurations):
                if not res is None:
                    rows.append(res)
                pbar.update()

    # store matrix
    if not fname is None:
        with open(fname, 'wb') as fd:
            pickle.dump({
                'data': rows
            }, fd)
    else:
        return rows

def generate_random_data(fname, paramter_shift=10):
    """ Generate random data for comparison with experimental one
    """
    def gen_rand_mat(dim=3):
        """ Generate random correlation matrix
        """
        tmp = npr.uniform(-1, 1, (dim,dim))

        # make matrix symmetric
        for i in range(dim):
            for j in range(i+1, dim):
                tmp[i,j] = tmp[j,i]

        return tmp

    def handle_random_case(size):
        random_raw = None, gen_rand_mat(), None

        row = []
        for _ in range(size):
            random_enh = None, gen_rand_mat(4), None
            row.append((None, random_enh))

        return [(None, random_raw), row]

    # generate random data
    rows = []
    for _ in trange(paramter_shift**2):
        rows.append(handle_random_case(64))

    # store matrix
    with open(fname, 'wb') as fd:
        pickle.dump({
            'data': rows,
            'corr_stdev': np.std([r[0][1][1] for r in rows], axis=0)
        }, fd)

def main():
    """
    Main interface
    """
    if len(sys.argv) != 2:
        print('Usage: %s <data file>' % sys.argv[0])
        exit(-1)

    generate_data(sys.argv[1], gen_func=generate_basic_system)
    #generate_random_data(sys.argv[1])
    #generate_motif_data(sys.argv[1])

if __name__ == '__main__':
    main()
