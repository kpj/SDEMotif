"""
Investigate 3+1 node network with varied parameters
"""

import sys
import copy
import pickle
import itertools

import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import tqdm

from system import SDESystem
from setup import generate_basic_system
from main import analyze_system
from utils import extract_sig_entries
from plotter import save_figure, plot_system, plot_corr_mat, plot_system_evolution


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

    # sort by network density
    def get_density(syst):
        max_edge_num = syst.jacobian.shape[0] * (syst.jacobian.shape[0]+1)
        return np.count_nonzero(syst.jacobian) / max_edge_num
    systems = sorted(systems, key=get_density)

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
            ] # rows in output plot
        }
    """
    param_range = np.linspace(0.1, 5, paramter_shift)

    # generate data
    rows = []
    for k_m in tqdm(param_range):
        for k_23 in tqdm(param_range, nested=True):
            syst = generate_basic_system(k_m=k_m, k_23=k_23)
            more = add_node_to_system(syst)

            res = handle_systems(syst, more)
            rows.append(res)

    # order rows by absolute jacobian mean
    def get_parameter_sort(row):
        raw_res, _ = row
        raw, _, _ = raw_res
        return np.mean(abs(raw.jacobian))
    rows = sorted(rows, key=get_parameter_sort)

    # store matrix
    with open(fname, 'wb') as fd:
        pickle.dump({
            'data': rows
        }, fd)

def sort_columns(data, sort_data, sort_functions):
    """ Sort columns of `data` by multiple sort functions applied to `sort_data` in order
    """
    tmp = np.transpose(data).tolist()
    sort_tmp = np.copy(sort_data)
    for sfunc in sort_functions[::-1]:
        tmp = [x for y, x in sorted(
            zip(sort_tmp, tmp), key=lambda pair: sfunc(pair[0]))]
        sort_tmp = list(sorted(sort_tmp, key=sfunc))
    return np.transpose(tmp)

def preprocess_data(data, val_func, sort_functions):
    """ Extract data information.
        Sort columns primarily by first sort_function and then the others in order
    """
    # compute matrix entries
    def handle_enh_entry(raw_res, enh_res):
        raw, raw_mat, raw_sol = raw_res
        enh, enh_mat, enh_sol = enh_res

        if raw_mat is None or enh_mat is None:
            return -1

        enh_mat = enh_mat[:-1,:-1] # disregard fourth node
        raw_vals = extract_sig_entries(raw_mat)
        enh_vals = extract_sig_entries(enh_mat)

        return val_func(raw_vals, enh_vals)

    plot_data = []
    for raw, enh_res in data: # for each row
        plot_data.append([handle_enh_entry(raw, enh) for enh in enh_res])

    # order columns by `sort_func`
    char_netws = [n[0] for n in data[0][1]]
    plot_data = sort_columns(plot_data, char_netws, sort_functions)

    # generate axes labels
    xtick_labels = sorted([sort_functions[0](n) for n in char_netws])
    ytick_labels = [round(np.mean(abs(r[0].jacobian)), 2) for r, e in data]

    return plot_data, xtick_labels, ytick_labels

def plot_result(inp, vfunc, sfuncs, title, fname):
    """ Plot generated matrix
    """
    # preprocess data
    data, xticks, yticks = preprocess_data(inp['data'], vfunc, sfuncs)

    # create plot
    plt.xticks(np.arange(len(data[0]), dtype=np.int), xticks)
    plt.yticks(np.arange(len(data), dtype=np.int), yticks)

    plt.setp(plt.gca().get_xticklabels(), fontsize=4, rotation='vertical')
    plt.setp(plt.gca().get_yticklabels(), fontsize=4)

    plt.tick_params(
        axis='both', which='both', labelleft='on',
        bottom='off', top='off', labelbottom='on', left='off', right='off')

    plt.title(title)
    plt.xlabel(sfuncs[0].__doc__)
    plt.ylabel('absolute mean of Jacobian')

    col_list = [(0.7,0.7,0.7), (0,0,1), (1,0,0)]
    cmap = mpl.colors.ListedColormap(col_list, name='highlighter')
    cmap.set_under('white')

    plt.imshow(
        data,
        interpolation='nearest', cmap=cmap,
        vmin=0, vmax=np.max(data))
    plt.colorbar(ticks=range(np.max(data)+1), extend='min')

    save_figure(fname, bbox_inches='tight')
    plt.close()

    plot_individuals(inp['data'], data, fname)

def plot_individuals(data, mat, fname, num=3):
    """ Plot a selection of individual results
    """
    # select "best" examples for networks
    scores = []
    for col in mat.T:
        scores.append(col[col > 0].sum())
    xsel = np.argsort(-np.array(scores))[:num]

    ysel = []
    for col in mat.T[xsel]:
        csel = np.argsort(col)[-1]
        ysel.append(csel)

    netws = []
    for x, y in zip(xsel, ysel):
        raw = data[y][0]
        cur = data[y][1][x]
        netws.append((raw, cur))

    # plot selected networks
    fig = plt.figure(figsize=(25, 4*len(netws)))
    gs = mpl.gridspec.GridSpec(len(netws), 6, width_ratios=[1, 1, 2, 1, 1, 2])

    counter = 0
    for i, net in enumerate(netws):
        raw, enh = net
        if raw[1] is None or enh[1] is None:
            counter += 1
            continue

        plot_system(raw[0], plt.subplot(gs[i, 0]))
        plot_corr_mat(raw[1], plt.subplot(gs[i, 1]))
        plot_system_evolution(raw[2], plt.subplot(gs[i, 2]))

        plot_system(enh[0], plt.subplot(gs[i, 3]))
        plot_corr_mat(enh[1], plt.subplot(gs[i, 4]))
        plot_system_evolution(enh[2], plt.subplot(gs[i, 5]))

    if counter > 0:
        print('{} broken results'.format(counter))

    save_figure('%s_zoom.pdf' % fname.replace('.pdf', ''), bbox_inches='tight', dpi=300)
    plt.close()

def handle_plots(inp):
    """ Generate plots for varying data extraction functions
    """
    # value functions
    def annihilate_low_correlations(vals, threshold=0.1):
        """ Take care of small fluctuations around 0
        """
        vals[abs(vals) <= threshold] = 0
        return vals

    def get_sign_changes(raw_vals, enh_vals):
        """ Compute number of sign changes
        """
        raw_vals = annihilate_low_correlations(raw_vals)
        enh_vals = annihilate_low_correlations(enh_vals)
        return np.sum(np.invert(np.sign(raw_vals) == np.sign(enh_vals)))

    def get_rank_changes(raw_vals, enh_vals):
        """ Detect changes in the order of correlations
        """
        raw_vals = annihilate_low_correlations(raw_vals)
        enh_vals = annihilate_low_correlations(enh_vals)
        return np.sum(np.invert(np.argsort(raw_vals) == np.argsort(enh_vals)))

    # sorting functions
    def sort_by_network_density(netw):
        """network density"""
        edge_num = np.count_nonzero(netw.jacobian)
        max_edge_num = netw.jacobian.shape[0]**2
        return round(edge_num / max_edge_num, 2)

    def sort_by_indeg(netw):
        """in-degree of fourth node"""
        in_vec = netw.jacobian[:,-1][:-1]
        return np.sum(in_vec)

    def sort_by_outdeg(netw):
        """out-degree of fourth node"""
        out_vec = netw.jacobian[-1,:][:-1]
        return np.sum(out_vec)

    def sort_by_cycle_num(netw):
        """number of cycles"""
        graph = nx.from_numpy_matrix(netw.jacobian, create_using=nx.DiGraph())
        return len(list(nx.simple_cycles(graph)))

    # plots
    for vfunc, title in zip([get_sign_changes, get_rank_changes], ['sign', 'rank']):
        ptitle = '{} changes'.format(title)

        plot_result(inp,
            vfunc, [sort_by_network_density],
            ptitle, 'images/matrix_{}_netdens.pdf'.format(title))
        plot_result(inp,
            vfunc, [sort_by_indeg, sort_by_outdeg],
            ptitle, 'images/matrix_{}_indeg.pdf'.format(title))
        plot_result(inp,
            vfunc, [sort_by_outdeg, sort_by_indeg],
            ptitle, 'images/matrix_{}_outdeg.pdf'.format(title))
        plot_result(inp,
            vfunc, [sort_by_cycle_num],
            ptitle, 'images/matrix_{}_cycles.pdf'.format(title))


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
