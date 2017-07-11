"""
Clean pipeline of robustness detection
"""

import os
import sys
import copy
import pickle
from typing import Any, Tuple, List, Callable
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as scis

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm, trange

from solver import solve_system
from filters import filter_steady_state
from nm_data_generator import add_node_to_system
from setup import generate_basic_system, generate_two_node_system, generate_v_out, generate_motifs


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
        ode_system = copy.deepcopy(sde_system)
        ode_system.fluctuation_vector = np.zeros(sde_system.fluctuation_vector.shape)

        corr_mats = []
        for _ in trange(reps):
            sde_sol = solve_system(sde_system)
            ode_sol = solve_system(ode_system)

            sol = ode_sol - sde_sol
            sol_extract = sol.T[int(len(sol.T)*3/4):] # extract steady-state

            # filter based on ODE solution
            if filter_steady_state(ode_sol.T[int(len(ode_sol.T)*3/4):]):
                continue

            # compute correlations
            stop = False
            dim = sol_extract.shape[1]
            mat = np.empty((dim,dim))
            for i in range(dim):
                for j in range(dim):
                    xs, ys = sol_extract[:,i], sol_extract[:,j]
                    try:
                        cc, pval = scis.pearsonr(xs, ys)
                        mat[i,j] = cc
                    except FloatingPointError:
                        stop = True
                        break
                if stop:
                    break

            if not stop:
                corr_mats.append(mat)
        return np.asarray(corr_mats)

    curs = []
    for enh in tqdm(enhanced):
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
        resolution = max(1, int(cpu_count() * 1/8))
        with Pool(resolution) as p:
            for res in p.starmap(simulate_systems, configurations):
                if not res is None:
                    data.append(res)
                pbar.update()

    # store data
    if not fname is None:
        with open(fname, 'wb') as fd:
            pickle.dump({
                'data': data,
                'motif': gen_func()
            }, fd)
    else:
        return data

def generate_system_data(motif_idx, motifs_three):
    """ Generate data for a given system
    """
    res = []
    # iterate over three ways of driving given motif
    for i, motif_func in enumerate(motifs_three):
        cur = generate_data(None, gen_func=motif_func)
        res.append(cur)
    return ({
        'motif': motif_func(),
        'idx': motif_idx
    }, res)

def generate_motif_data(prefix):
    """ Generate data for all motifs
    """
    motifs = generate_motifs()
    with tqdm(total=len(motifs)) as pbar:
        resolution = max(1, int(cpu_count() * 1/8))
        with ThreadPool(resolution) as p:
            # iterate over motif topologies
            for info, result in p.starmap(generate_system_data, zip(range(len(motifs)), motifs)):
                fname = '{}_{}'.format(prefix, info['idx'])
                with open(fname, 'wb') as fd:
                    pickle.dump({
                        'data': result,
                        'motif': info['motif']
                    }, fd)
                pbar.update()

def handle_enh_entry(entry, thres: float) -> List:
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

    return cur

def threshold_influence(
    data: List,
    ax=None, resolution: int = 100,
    fname_app: str = ''
) -> None:
    """ Plot robustness for varying threshold levels
    """
    threshold_list = np.logspace(-5, 0, resolution)

    # generate data
    fname_cache = f'cache/ti_data{fname_app}.csv'

    if not os.path.exists(fname_cache):
        tmp = {'threshold': [], 'correlation_transfer': [], 'param_config': []}
        for thres in tqdm(threshold_list):
            for i, entry in enumerate(data): # iterate over parameter configurations
                res = handle_enh_entry(entry, thres)
                if len(res) == 0:
                    continue

                for j, ct in enumerate(res):
                    tmp['threshold'].append(thres)
                    tmp['correlation_transfer'].append(ct)
                    tmp['param_config'].append(f'{i},{j}')

        df = pd.DataFrame(tmp)
        df.to_csv(fname_cache)
    else:
        print('Cached', fname_cache)
        df = pd.read_csv(fname_cache, index_col=0)

    if df.empty:
        return

    # normalize robustness values
    max_rob = df.groupby('threshold').mean()['correlation_transfer'].max()
    assert max_rob >= 0, df.head()
    if max_rob > 0:
        df['correlation_transfer'] /= max_rob

    print(df.describe())

    # data for histograms
    robust_vals, thres_vals = fixed_threshold(data)
    assert robust_vals.shape == thres_vals.shape
    if len(robust_vals) == 0:
        return

    if max_rob > 0:
        robust_vals /= max_rob

    # plot data
    if ax is None:
        plt.figure()
        ax_new = plt.gca()
    else:
        ax_new = ax

    atx = ax_new.twinx()
    aty = ax_new.twiny()

    sns.distplot(
        thres_vals, ax=atx,
        hist=False, kde=len(thres_vals)>1, rug=len(thres_vals)>1,
        color='k', kde_kws=dict(alpha=.1), rug_kws=dict(alpha=.2))
    if sum(robust_vals) > 0:
        sns.distplot(
            robust_vals, ax=aty, vertical=True,
            hist=False, kde=len(robust_vals)>1, rug=len(robust_vals)>1,
            color='k', kde_kws=dict(alpha=.1), rug_kws=dict(alpha=.2))
    ax_new.scatter(thres_vals, robust_vals, color='k', alpha=.05)

    sns.tsplot(
        df, ax=ax_new,
        time='threshold', unit='param_config', value='correlation_transfer')

    ax_new.set_xscale('log')
    ax_new.set_ylim((0, 1))
    ax_new.set_title(f'#entries: {df.shape[0]}')

    atx.set(yticklabels=[])
    aty.set(xticklabels=[])

    if ax is None:
        plt.savefig(f'images/threshold_influence{fname_app}.pdf')

    return df, {'thresholds': thres_vals, 'corr_trans': robust_vals}

def fixed_threshold(data: List) -> Tuple[Any, Any]:
    """ Compute robustness with `thres = \sigma / 2` and return values
    """
    robs, thresholds = [], []
    for entry in data:
        raw_corr_mats = entry['raw_corr_mats']
        if len(raw_corr_mats) == 0:
            continue
        if np.isnan(raw_corr_mats).any():
            continue

        # find \sigma of distribution with lowest \mu
        abs_corr_avg = abs(np.mean(raw_corr_mats, axis=0))
        min_idx = np.unravel_index(abs_corr_avg.argmin(), abs_corr_avg.shape)
        series = raw_corr_mats[:,min_idx[0],min_idx[1]] # better indexing?
        sigmah = np.std(series) / 2
        thresholds.append(sigmah)

        # handle entry
        res = handle_enh_entry(entry, sigmah)
        robs.append(np.mean(res))
    return np.asarray(robs), np.asarray(thresholds)

def motif_overview(prefix):
    """ Conduct analysis over range of motifs
    """
    # get data
    data = {}
    pref_dir = os.path.dirname(prefix)
    for fn in tqdm(os.listdir(pref_dir)):
        if fn.startswith(os.path.basename(prefix)):
            fname = os.path.join(pref_dir, fn)

            with open(fname, 'rb') as fd:
                inp = pickle.load(fd)

            data[fn] = {
                'idx': int(fn.split('_')[-1]),
                'motif': inp['motif'],
                'inp': inp
            }

    # plot data
    plt.figure(figsize=(6*len(data),13))
    gs = gridspec.GridSpec(9, len(data))

    # add motif and threshold plots
    df_stats_list = []
    for i, k in enumerate(sorted(data, key=lambda k: data[k]['idx'])):
        print('>', k)

        # motif
        a = plt.subplot(gs[:3,i])
        graph = nx.from_numpy_matrix(
            data[k]['motif'].jacobian.T, create_using=nx.DiGraph())
        pos = nx.circular_layout(graph)
        nx.draw(
            graph, pos, ax=a, node_size=60,
            with_labels=True, font_size=4)
        a.axis('on')
        a.set_xticks([], [])
        a.set_yticks([], [])
        a.set_title(data[k]['idx'])

        # threshold
        a = plt.subplot(gs[3:6,i])

        df_list = []
        extra_info = {}
        for j, rows in enumerate(data[k]['inp']['data']):
            if len(rows) == 0:
                continue

            res = threshold_influence(rows, ax=a)
            if res is None:
                continue

            df, extra = res
            df['run_id'] = j
            df_list.append(df)

            assert j not in extra_info
            extra_info[j] = extra

        if len(df_list) > 0:
            df_all = pd.concat(df_list)
        else:
            df_all = None

        a.tick_params(labelsize=6)
        a.xaxis.label.set_size(4)
        a.yaxis.label.set_size(4)
        a.title.set_size(4)

        # exemplary trajectory
        syst = copy.deepcopy(data[k]['motif'])
        assert syst.jacobian.shape == (3,3), syst

        def find_solution(sde_syst):
            # find fitting Jacobian
            cur_m = generate_motifs()[data[k]['idx']][0]
            param_range = np.linspace(1, 8, 10)

            # generate data
            configurations = []
            for k_m in param_range:
                for k_23 in param_range:
                    tmp = cur_m(k_m=k_m/2, k_23=k_23/2)
                    sde_syst.jacobian = tmp.jacobian

                    sde_sol = solve_system(sde_syst)

                    ode_syst = copy.deepcopy(sde_syst)
                    ode_syst.fluctuation_vector = np.array([0, 0, 0])
                    ode_syst.external_influence = np.array([0, 0, 0])
                    ode_sol = solve_system(ode_syst)

                    sol = ode_sol - sde_sol
                    sol_extract = ode_sol.T[int(len(ode_sol.T)*3/4):]

                    if not filter_steady_state(sol_extract):
                        return sol

            return None

        for j in range(3):
            # do simulation
            syst.fluctuation_vector = np.array([0, 0, 0])
            syst.fluctuation_vector[j] = 1
            syst.external_influence = np.array([0, 0, 0])
            syst.external_influence[j] = 5

            sol = find_solution(syst)
            if sol is None:
                continue

            # plot result
            ax = plt.subplot(gs[6+j,i])
            ax.tick_params(
                axis='both', which='both', labelleft='off',
                bottom='off', top='off', labelbottom='off', left='off', right='off')

            ax.plot(sol[0],
                ls='solid',
                marker='$0$', markevery=100,
                label=rf'{j} driven')
            ax.plot(sol[1],
                ls='dashed',
                marker='$1$', markevery=100)
            ax.plot(sol[2],
                ls='dotted',
                marker='$2$', markevery=100)

            ax.legend(loc='best', prop={'size':8})

        # robustness quantification
        values = {'data': [], 'run_id': [], 'type': []}
        if df_all is not None:
            for gid, group in df_all.groupby('run_id'):
                cur = group.sort_values('threshold')

                # mean correlation transfer
                values['run_id'].append(gid)
                values['data'].append(np.mean(extra_info[gid]['corr_trans']))
                values['type'].append('corr_trans')

                # AUC
                mean_thres = np.mean(extra_info[gid]['thresholds'])
                cur_sub = cur[cur['threshold']>mean_thres]

                t_vals = cur_sub['threshold'].tolist()
                m_vals = cur_sub['correlation_transfer'].tolist()
                area = np.trapz(m_vals, x=t_vals)

                values['run_id'].append(gid)
                values['data'].append(area)
                values['type'].append('area')

        df_stats = pd.DataFrame(values)
        df_stats['motif_idx'] = i
        df_stats_list.append(df_stats)

    plt.tight_layout()
    plt.savefig('images/motifs.pdf')

    return pd.concat(df_stats_list)

def plot_motif_statistics(df):
    """ Plot statistics over all motifs
    """
    plt.figure()

    plt.subplot(121)
    df_area = df[df['type']=='area'].groupby('motif_idx').max().reset_index()
    sns.barplot(x='motif_idx', y='data', data=df_area)
    plt.title('area')

    plt.subplot(122)
    df_ct = df[df['type']=='corr_trans'].groupby('motif_idx').max().reset_index()
    sns.barplot(x='motif_idx', y='data', data=df_ct)
    plt.title('corr_trans')

    plt.savefig('images/motif_statistics.pdf')


def main(fname) -> None:
    with open(fname, 'rb') as fd:
        inp = pickle.load(fd)

    fapp = f'__{os.path.basename(fname.replace(".", "_"))}'
    threshold_influence(np.asarray(inp['data']), fname_app=fapp)

if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    if len(sys.argv) == 1:
        #initial_tests()

        generate_data(
            'results/new_data_ffl.dat', gen_func=generate_basic_system)
        generate_data(
            'results/new_data_vout.dat', gen_func=generate_v_out)
        generate_data(
            'results/new_data_link.dat', gen_func=generate_two_node_system)

        generate_motif_data('results/new_data_motifs.dat')
    elif len(sys.argv) == 2:
        if os.path.exists(sys.argv[1]):
            main(sys.argv[1])
        else:
            # assume it's a motif prefix
            df = motif_overview(sys.argv[1])

            df.to_csv('results/motif_statistics.csv')
            plot_motif_statistics(df)
    else:
        print('Usage: {} [data file]'.format(sys.argv[0]))
        exit(-1)
