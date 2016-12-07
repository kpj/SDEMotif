"""
Produce some nice figures
"""

from itertools import cycle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from setup import generate_basic_system
from plotter import plot_system, plot_system_evolution, plot_corr_mat
from nm_data_generator import add_node_to_system
from main import analyze_system
from solver import solve_system
from filters import filter_steady_state
from utils import compute_correlation_matrix


def detailed_system():
    syst = generate_basic_system()

    plt.figure()
    plot_system(syst, plt.gca())
    plt.savefig('presentation/images/FFL.pdf', dpi=300)

def plot_series(syst, ax, uosd=True):
    syst, mat, sol = analyze_system(
        syst, use_ode_sde_diff=uosd,
        repetition_num=5, tmax=10)
    plot_system_evolution(sol[:50], ax, show_legend=False)

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_xlabel('')
    ax.set_ylabel('')

def plot_mat(syst, ax):
    syst, mat, sol = analyze_system(
        syst, use_ode_sde_diff=True,
        repetition_num=5, tmax=10)
    plot_corr_mat(mat, ax)

    ax.set_xticks([], [])
    ax.set_yticks([], [])

def plot_hist(syst, ax):
    single_run_matrices = []
    for _ in range(50):
        sol = solve_system(syst)
        sol_extract = sol.T[int(len(sol.T)*3/4):]

        if filter_steady_state(sol_extract):
            continue

        single_run_mat = compute_correlation_matrix(np.array([sol_extract]))

        if single_run_mat.shape == (4, 4):
            single_run_mat = single_run_mat[:-1,:-1]
        assert single_run_mat.shape == (3, 3)

        single_run_matrices.append(single_run_mat)
    single_run_matrices = np.asarray(single_run_matrices)

    # plotting
    cols = cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k'])
    for i, row in enumerate(single_run_matrices.T):
        for j, series in enumerate(row):
            if i == j: break

            sns.distplot(series, ax=ax, label=r'$c_{{{},{}}}$'.format(i,j))

    ax.set_xlim((-1,1))
    ax.set_xticks([], [])
    ax.set_yticks([], [])

def configuration_overview(func, fname, draw_all=True):
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 5, width_ratios=[1,.2,1,1,1])

    for i, conf in enumerate([(1,1), (4,2), (2,1)]):
        syst = generate_basic_system(*conf)
        func(syst, plt.subplot(gs[i, 0]))

        if draw_all:
            for j, m in enumerate(add_node_to_system(syst)[3:6]):
                func(m, plt.subplot(gs[i, j+2]))

    if draw_all:
        fig.text(0.5, 0.04, 'varied embedding', ha='center', fontsize=20)
    fig.text(0.085, 0.5, 'varied parameters', va='center', rotation='vertical', fontsize=20)

    plt.savefig('presentation/images/overview_{}.pdf'.format(fname))

def distribution_filter_threshold():
    plt.figure()

    sns.distplot(
        np.random.normal(.1, .1, size=1000),
        label='before embedding', hist_kws=dict(alpha=.2))
    sns.distplot(
        np.random.normal(0, .1, size=1000),
        label='after embedding', hist_kws=dict(alpha=.2))

    thres = .1
    plt.axvspan(-thres, thres, facecolor='r', alpha=0.1, label='threshold')

    plt.xlim((-1,1))
    plt.xlabel('correlation')
    plt.legend(loc='best')

    plt.savefig('presentation/images/dist_thres.pdf')

def main():
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    detailed_system()
    configuration_overview(
        lambda s,a: plot_system(s, a, netx_plot=True),
        'sub', draw_all=False)
    configuration_overview(
        lambda s,a: plot_system(s, a, netx_plot=True), 'all')
    configuration_overview(
        lambda s,a: plot_series(s,a,uosd=False), 'series_orig')
    configuration_overview(plot_series, 'series')
    configuration_overview(plot_mat, 'mat')
    configuration_overview(plot_hist, 'hist')
    distribution_filter_threshold()

if __name__ == '__main__':
    main()
