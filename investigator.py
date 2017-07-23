"""
Investigate data in various ways
"""

import sys
import copy
import pickle
from itertools import cycle

import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import tqdm, trange

from solver import solve_system
from utils import extract_sig_entries, compute_correlation_matrix
from plotter import plot_histogram, plot_system, save_figure, plot_system_evolution
from setup import generate_basic_system
from main import analyze_system
from nm_data_generator import add_node_to_system
from filters import filter_steady_state


def plot_correlation_hist(data):
    """ Plot histogram of all correlations
    """
    # gather data
    corrs = []
    for raw_res, enh_list in data:
        _, raw_mat, _ = raw_res

        if not raw_mat is None:
            raw_vec = extract_sig_entries(raw_mat)
            corrs.extend(raw_vec)

        for enh_res in enh_list:
            _, enh_mat, _ = enh_res

            if not enh_mat is None:
                enh_vec = extract_sig_entries(enh_mat)
                corrs.extend(enh_vec)

    # plot result
    fig = plt.figure()

    plot_histogram(corrs, plt.gca())
    plt.xlabel('simulated correlations')

    fig.savefig('images/all_sim_corrs.pdf')

def check_ergodicity(reps=500):
    """ Check whether simulated systems are ergodic
    """
    def get_matrices(syst, entry_num=100):
        """ Get correlation matrices for both cases
        """
        # multiple entries from single run
        single_run_matrices = []
        for _ in range(entry_num):
            sol = solve_system(syst)

            extract = sol.T[-entry_num:]
            single_run_mat = compute_correlation_matrix(np.array([extract]))

            single_run_matrices.append(single_run_mat)
        avg_single_mat = np.mean(single_run_matrices, axis=0)

        # one entry from multiple runs
        multiple_runs = []
        for _ in range(entry_num):
            sol = solve_system(syst)

            extract = sol.T[-1].T
            multiple_runs.append(extract)
        multiple_mat = compute_correlation_matrix(np.array([multiple_runs]))

        return avg_single_mat, multiple_mat

    syst = generate_basic_system()

    single_runs = []
    multiple_runs = []
    for _ in trange(reps):
        sm, rm = get_matrices(syst)

        single_runs.append(sm)
        multiple_runs.append(rm)
    single_runs = np.array(single_runs)
    multiple_runs = np.array(multiple_runs)

    # plot result
    dim = syst.jacobian.shape[1]

    plt.figure(figsize=(6, 14))
    gs = mpl.gridspec.GridSpec(int((dim**2-dim)/2), 1)

    axc = 0
    for i in range(dim):
        for j in range(dim):
            if i == j: break
            ax = plt.subplot(gs[axc])

            plot_histogram(
                single_runs[:,i,j], ax,
                alpha=0.5,
                label='Multiple entries from single run')
            plot_histogram(multiple_runs[:,i,j], ax,
                facecolor='mediumturquoise', alpha=0.5,
                label='One entry from multiple runs')

            ax.set_title('Nodes {}, {}'.format(i, j))
            ax.set_xlabel('correlation')
            ax.legend(loc='best')

            axc += 1

    plt.tight_layout()
    plt.savefig('images/ergodicity_check.pdf')

def single_corr_coeff_hist(reps=200):
    """ Plot distribution of single correlation coefficients for various parameters
    """
    def do(gs, res):
        param_range = np.linspace(.1, 5, res)
        currow = 0
        for k_m in tqdm(param_range):
            for k_23 in tqdm(param_range):
                syst = generate_basic_system(k_m=k_m, k_23=k_23)

                single_run_matrices = []
                for r in trange(reps):
                    _,mat,sol = analyze_system(syst, repetition_num=1)
                    if mat is None:
                        continue

                    sol_extract = sol.T[int(len(sol.T)*3/4):]

                    if r == 0:
                        plot_system_evolution(
                            sol_extract.T,
                            plt.subplot(gs[currow,2]), show_legend=False)

                    single_run_mat = compute_correlation_matrix(np.array([sol_extract]))

                    if single_run_mat.shape == (4, 4):
                        single_run_mat = single_run_mat[:-1,:-1]
                    assert single_run_mat.shape == (3, 3)

                    single_run_matrices.append(single_run_mat)

                plot_system(syst, plt.subplot(gs[currow,0]))

                single_run_matrices = np.asarray(single_run_matrices)
                for i, row in enumerate(single_run_matrices.T):
                    for j, series in enumerate(row):
                        if i == j: break

                        ax = plt.subplot(gs[currow,1])
                        sns.distplot(series, ax=ax, label=r'$c_{{{},{}}}$'.format(i,j))
                        ax.set_xlim((-1,1))

                currow += 1

    # generate plots
    res = 3

    plt.figure(figsize=(20,30))
    gs = mpl.gridspec.GridSpec(res**2, 3, width_ratios=[1,1,2])

    sns.set_style('white')
    plt.style.use('seaborn-poster')

    do(gs, res)

    plt.tight_layout()
    save_figure('images/correlation_distribution.pdf', bbox_inches='tight')

def lyapunov_equation():
    """ Check if our experiments satisfy the lyapunov equation
    """
    # create systems
    sde_system = generate_basic_system()
    sde_system.fluctuation_vector[-1] = 2

    ode_system = copy.deepcopy(sde_system)
    ode_system.fluctuation_vector = np.zeros(sde_system.fluctuation_vector.shape)

    # generate data
    sde_sol = solve_system(sde_system)
    ode_sol = solve_system(ode_system)

    sol = ode_sol - sde_sol
    sol_extract = sol.T[int(len(sol.T)*3/4):] # extract steady-state

    # investigate result
    J = sde_system.jacobian
    C = np.cov(sol_extract.T)
    D = np.diag(sde_system.fluctuation_vector)

    term1 = J @ C + C @ J.T
    term2 = -2 * D

    print(term1, '\n',term2)

    # plot stuff
    #plt.plot(sol_extract)
    plt.scatter(term1.ravel(), term2.ravel())

    plt.title(f'Fluctuation vector: {sde_system.fluctuation_vector}')
    plt.xlabel('J @ C + C @ J.T')
    plt.ylabel('-2 * D')

    plt.savefig('images/lyapunov_equation.pdf')

def correlation_patterns():
    """ Investigate various correlation patterns
    """
    def read(fname):
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
            return np.asarray(inp['data'])

    def aggregate_corr_matrices(data):
        mats = []
        for entry in data: # for each parameter configuration
            raw_corr_mats = entry['raw_corr_mats']
            enh_corr_mat_list = entry['enh_corr_mat_list']

            for enh_corr_mats in enh_corr_mat_list: # for each embedding
                if enh_corr_mats.size == 0:
                    continue
                trans = enh_corr_mats[:,:3,:3]

                mats.extend(trans)
            mats.extend(raw_corr_mats)
        mats = np.asarray(mats)

        return mats

    # read data
    data_ffl = read('results/new_data_ffl.dat')
    data_vout = read('results/new_data_vout.dat')

    # convert data
    corr_ffl = aggregate_corr_matrices(data_ffl)
    corr_vout = aggregate_corr_matrices(data_vout)

    # plot data
    plt.figure()

    plt.subplot(211)
    plt.title('FFL')
    sns.distplot(corr_ffl[:,0,1], kde=False, label=r'$c_{12}$')
    sns.distplot(corr_ffl[:,0,2], kde=False, label=r'$c_{13}$')
    plt.legend(loc='best')

    plt.subplot(212)
    plt.title('Vout')
    sns.distplot(corr_vout[:,0,1], kde=False, label=r'$c_{12}$')
    sns.distplot(corr_vout[:,0,2], kde=False, label=r'$c_{13}$')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('images/correlation_patterns.pdf')

    plt.figure()
    plt.title(r'$c_{23}$')
    sns.distplot(corr_ffl[:,1,2], kde=False, label=r'FFL')
    sns.distplot(corr_vout[:,1,2], kde=False, label=r'Vout')
    plt.legend(loc='best')
    plt.savefig('images/correlation_patterns_2.pdf')


def main(data):
    """ Analyse data
    """
    #if data is None:
    #    check_ergodicity()
    #else:
    #    plot_correlation_hist(data)
    #single_corr_coeff_hist()
    #lyapunov_equation()
    correlation_patterns()

if __name__ == '__main__':
    main(np.load(sys.argv[1])['data'] if len(sys.argv) == 2 else None)
