"""
Clean pipeline of robustness detection
"""

import sys
import copy
import pickle
from typing import Tuple, List, Callable
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import scipy.stats as scis

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm, trange

from solver import solve_system
from filters import filter_steady_state
from nm_data_generator import add_node_to_system
from setup import generate_basic_system, generate_two_node_system


def threshold_div(distr: np.ndarray, thres: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Divide distribution into below `-thres` and above `thres`
    """
    below = distr[distr<-thres]
    above = distr[distr>thres]
    return below, above

def compare_distributions(dis1: np.ndarray, dis2: np.ndarray, thres: float) -> float:
    """ Compare two distributions according to robustness measure
    """
    c_1m, c_1p = threshold_div(dis1, thres)
    c_2m, c_2p = threshold_div(dis2, thres)

    n_1m, n_1p = c_1m.size/dis1.size, c_1p.size/dis1.size
    n_2m, n_2p = c_2m.size/dis2.size, c_2p.size/dis2.size

    return n_1p * n_2m * (n_2m - n_1m) + n_1m * n_2p * (n_2p - n_1p)

def initial_tests(n: int = 1000) -> None:
    """ Conduct some initial tests with robustness measure
    """
    get_dis = lambda x: np.random.normal(x, 0.1, size=n)

    thresholds = np.logspace(-4, -1, 10)
    x_space = np.linspace(-1, 1, 100)

    plt.figure()
    gs = gridspec.GridSpec(2, 2)

    for i, dis_3_x in enumerate([-.08, 0.5]):
        dis_3 = get_dis(dis_3_x)

        # generate data
        data = {t: [] for t in thresholds}
        data.update({'x': []})
        for x in tqdm(x_space):
            dis_4 = get_dis(x)

            data['x'].append(x)
            for t in thresholds:
                rob = compare_distributions(dis_3, dis_4, t)
                data[t].append(rob)

        df = pd.DataFrame(data)
        df.set_index('x', inplace=True)

        # plot data
        ax = plt.subplot(gs[i, 1])
        df.plot(legend=False, ax=ax)

        ax.set_title(r'First distribution: $x={}$'.format(dis_3_x))
        ax.set_xlabel(r'$\mu$ of second distribution')
        ax.set_ylabel('magic quantity')

        ax.set_xlim((-1,1))
        ax.set_ylim((0,1))

        # plot shift overview
        ax = plt.subplot(gs[i, 0])

        sns.distplot(
            dis_3, hist_kws=dict(alpha=.2),
            ax=ax)
        for x in x_space[::24]:
            dis_4 = get_dis(x)
            sns.distplot(
                dis_4, hist_kws=dict(alpha=.2), kde_kws=dict(alpha=.2),
                ax=ax, color='gray')

        ax.set_xlim((-1,1))

    plt.tight_layout()
    plt.savefig('images/robustness_development.pdf')

def simulate_systems(raw, enhanced, reps=100):
    """ Simulate given systems and return raw vs enhanced versions
    """
    def sim(sde_system):
        ode_system = copy.copy(sde_system)
        ode_system.fluctuation_vector = np.zeros(sde_system.fluctuation_vector.shape)

        corr_mats = []
        for _ in range(reps):
            sde_sol = solve_system(sde_system)
            ode_sol = solve_system(ode_system)

            sol = ode_sol - sde_sol
            sol_extract = sol.T[int(len(sol.T)*3/4):] # extract steady-state

            if filter_steady_state(ode_sol.T[int(len(ode_sol.T)*3/4):]):
                continue

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

    curs = []
    for enh in enhanced:
        curs.append(sim(enh))

    return {
        'raw_corr_mats': sim(raw),
        'enh_corr_mat_list': curs #[sim(enh) for enh in enhanced]
    }

def generate_data(
    fname: str, gen_func: Callable = generate_basic_system,
    paramter_shift: int = 10
) -> None:
    """ Generate individual correlation realizations for varying parameters and embeddings
    """
    param_range = np.linspace(1, 8, paramter_shift)

    # generate data
    configurations = []
    for k_m in param_range:
        for k_23 in param_range:
            syst = gen_func(k_m=k_m/2, k_23=k_23/2)
            more = add_node_to_system(syst)
            configurations.append((syst, more))

    # simulate data
    data = []
    with tqdm(total=len(configurations)) as pbar:
        resolution = int(cpu_count() * 3/4)
        with Pool(resolution) as p:
            for res in p.starmap(simulate_systems, configurations):
                if not res is None:
                    data.append(res)
                pbar.update()

    # store data
    with open(fname, 'wb') as fd:
        pickle.dump({
            'data': data
        }, fd)

def handle_enh_entry(entry, thres: float) -> float:
    """ Compare distribution from given simulation results.
        This handles an entry which contains all embeddings per parameter config
    """
    raw_corr_mats = entry['raw_corr_mats']
    enh_corr_mat_list = entry['enh_corr_mat_list']

    dim = raw_corr_mats.shape[-1]
    coords = np.tril_indices(dim, k=-1)

    cur = []
    for enh_corr_mats in enh_corr_mat_list:
        if enh_corr_mats.size == 0:
            continue

        for i,j in zip(*coords):
            dis_3 = raw_corr_mats[:,i,j]
            dis_4 = enh_corr_mats[:,i,j]
            rob = compare_distributions(dis_3, dis_4, thres)
            cur.append(rob)

    return np.mean(cur)

def threshold_influence(data: List, resolution: int = 100) -> None:
    """ Plot robustness for varying threshold levels
    """
    threshold_list = np.logspace(-5, 0, resolution-1)

    # generate data
    df = pd.DataFrame()
    for thres in tqdm(threshold_list):
        for i, entry in enumerate(data): # iterate over parameter configurations
            res = handle_enh_entry(entry, thres)
            df = df.append({
                'threshold': thres,
                'robustness': res,
                'param_config': i
            }, ignore_index=True)

    print(df.describe())

    # plot data
    plt.figure()

    sns.tsplot(
        df,
        time='threshold', unit='param_config', value='robustness')
    plt.xscale('log')

    plt.savefig('images/threshold_influence.pdf')

def fixed_threshold(data: List) -> float:
    """ Compute robustness with `thres = \sigma / 2`
    """
    robs = []
    for entry in data:
        raw_corr_mats = entry['raw_corr_mats']

        # find \sigma of distribution with lowest \mu
        abs_corr_avg = abs(np.mean(raw_corr_mats, axis=0))
        min_idx = np.unravel_index(abs_corr_avg.argmin(), abs_corr_avg.shape)
        series = raw_corr_mats[:,min_idx[0],min_idx[1]] # better indexing?
        sigmah = np.std(series) / 2

        # handle entry
        res = handle_enh_entry(entry, sigmah)
        robs.append(res)
    return np.mean(robs)

def main(fname) -> None:
    with open(fname, 'rb') as fd:
        inp = pickle.load(fd)

    print(fixed_threshold(np.asarray(inp['data'])))
    threshold_influence(np.asarray(inp['data']))

if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    if len(sys.argv) == 1:
        #initial_tests()

        generate_data(
            'results/new_data_ffl.dat', gen_func=generate_basic_system)
        generate_data(
            'results/new_data_link.dat', gen_func=generate_two_node_system)
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Usage: {} [data file]'.format(sys.argv[0]))
        exit(-1)
