"""
Investigate 3+1 node network with varied parameters
"""

import sys
import copy
import pickle
import itertools

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram

import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import tqdm

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
    raw_res = analyze_system(raw, filter_mask=[3])
    if raw_res[1] is None:
        return None

    row = []
    for enh in enhanced:
        enh_res = analyze_system(enh, filter_mask=[3])
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
            if not res is None:
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
    inds = range(len(tmp))
    for sfunc in sort_functions[::-1]:
        tmp = [x for y, x in sorted(
            zip(sort_tmp, tmp), key=lambda pair: sfunc(pair[0]))]
        inds = [x for y, x in sorted(
            zip(sort_tmp, inds), key=lambda pair: sfunc(pair[0]))]
        sort_tmp = list(sorted(sort_tmp, key=sfunc))
    return np.transpose(tmp), inds

def preprocess_data(data, val_func, sort_functionality):
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
    plot_data = np.array(plot_data)

    # sort rows/columns in miraculous ways
    xtick_func = None
    repos = None
    char_netws = [n[0] for n in data[0][1]]

    if isinstance(sort_functionality, list):
        plot_data, repos = sort_columns(
            plot_data, char_netws, sort_functionality)

        xtick_func = sort_functionality[0]
    elif isinstance(sort_functionality, tuple):
        xtick_func, _ = sort_functionality
        repos = range(len(plot_data[0]))
    else:
        raise RuntimeError(
            'Invalid sort-method ({})'.format(sort_functionality))

    # generate axes labels
    xtick_labels = np.array([xtick_func(n) for n in char_netws])[repos]
    ytick_labels = [round(np.mean(abs(r[0].jacobian)), 2) for r, e in data]

    return plot_data, xtick_labels, ytick_labels

def cluster_data(mat, metric):
    """ Cluster given data
    """
    link = linkage(mat.T, metric=metric)

    dendr = dendrogram(
        link, no_plot=True)

    pos = dendr['leaves']
    return mat.T[pos].T, pos

def plot_result(inp, vfunc, sfuncs, title, fname):
    """ Plot generated matrix

        `sfuncs` can either be a list of functions or a string of the form:
            * cluster:euclidean
            * cluster:hamming
        (generally every metric for scipy.spatial.distance.pdist)
    """
    print('Plotting "{}"'.format(fname), end='... ', flush=True)

    # preprocess data
    data, xticks, yticks = preprocess_data(inp['data'], vfunc, sfuncs)

    # stop, it's plotting time!
    if isinstance(sfuncs, tuple): # there will be clustering
        xtick_func, spec = sfuncs
        metr = spec.split(':')[1]

        # remove noisy signals for clustering
        data[data < 0] = 0

        dat = []
        for i, row in enumerate(data):
            dat.append((yticks[i], row))
        df = pd.DataFrame.from_items(dat, columns=xticks, orient='index')

        col_list = [(.7,.7,.7), (0,0,1), (1,0,0)]
        cmap = mpl.colors.ListedColormap(col_list, name='highlighter')
        cmap.set_under('white')

        plt.figure()
        cg = sns.clustermap(
            df, cmap=cmap, vmin=0, vmax=np.max(data),
            row_cluster=False, metric=metr)

        plt.setp(cg.ax_heatmap.xaxis.get_ticklabels(), rotation=90, size=6)
        plt.setp(cg.ax_heatmap.yaxis.get_ticklabels(), rotation=0, size=5)

        save_figure(fname, bbox_inches='tight')
        plt.close()
    else:
        # "normal" plot
        mpl.style.use('default') # possibly reset seaborn styles

        plt.figure()

        plt.xticks(np.arange(len(data[0]), dtype=np.int), xticks)
        plt.yticks(np.arange(len(data), dtype=np.int), yticks)

        plt.setp(plt.gca().get_xticklabels(), fontsize=3, rotation='vertical')
        plt.setp(plt.gca().get_yticklabels(), fontsize=3)

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

        # mark "zoomed" columns
        sel_one, netws_one = select_column_by_jacobian(inp['data'], np.array([
            [1,0,0,1],
            [1,1,0,0],
            [1,1,1,0],
            [0,0,1,1]
        ]))
        sel_two, netws_two = select_column_by_jacobian(inp['data'], np.array([
            [1,0,0,1],
            [1,1,0,0],
            [1,1,1,0],
            [1,0,0,1]
        ]))

        sel_xticks = [item for item in plt.gca().get_xticklabels()]
        sel_xticks[sel_one].set_weight('bold')
        sel_xticks[sel_two].set_weight('bold')
        plt.gca().set_xticklabels(sel_xticks)

        # mark "zoomed" rows
        sel_blue, netws_blue = select_row_by_count(inp['data'], data, 1)
        sel_red, netws_red = select_row_by_count(inp['data'], data, 2)

        sel_yticks = [item for item in plt.gca().get_yticklabels()]
        sel_yticks[sel_blue].set_weight('bold')
        sel_yticks[sel_red].set_weight('bold')
        plt.gca().set_yticklabels(sel_yticks)

        # save figure
        save_figure(fname, bbox_inches='tight')

        # plot best examples
        plot_individuals(netws_one, '{}_col_one'.format(fname))
        plot_individuals(netws_two, '{}_col_two'.format(fname))

        plot_individuals(netws_blue, '{}_row_blue'.format(fname))
        plot_individuals(netws_red, '{}_row_red'.format(fname))

    print('Done')

def select_column_by_jacobian(data, jac):
    """ Select column by approximated jacobian
    """
    ind = None
    for i, (syst, mat, sol) in enumerate(data[0][1]):
        nz = np.nonzero(syst.jacobian)
        comp_jac = np.zeros_like(syst.jacobian, dtype=int)
        comp_jac[nz] = 1

        if (comp_jac == jac).all():
            ind = i
            break

    res = []
    if ind is None:
        raise RuntimeError('No match found')
    else:
        for i in range(len(data)):
            net = data[i][1][ind]
            res.append(net)

    return ind, res

def select_row_by_count(data, mat, pat):
    """ Count occurences of `pat` in row
    """
    # find matching row
    counts = []
    for row in mat:
        counts.append(row.tolist().count(pat))
    row_sel = np.argsort(-np.array(counts))[:1]

    # collect respective networks
    netws = [data[row_sel][0]]
    netws.extend(data[row_sel][1])

    return row_sel, netws

def plot_individuals(examples, fname):
    """ Plot a selection of individual results
    """
    # plot selected networks
    fig = plt.figure(figsize=(25, 4*len(examples)))
    gs = mpl.gridspec.GridSpec(len(examples), 3, width_ratios=[1, 1, 2])

    counter = 0
    for i, net in enumerate(examples):
        if net[1] is None:
            counter += 1
            plot_system(net[0], plt.subplot(gs[i, 0]))
            plot_system_evolution(net[2], plt.subplot(gs[i, 2]))
            continue

        plot_system(net[0], plt.subplot(gs[i, 0]))
        plot_corr_mat(net[1], plt.subplot(gs[i, 1]))
        plot_system_evolution(net[2], plt.subplot(gs[i, 2]))

    if counter > 0:
        #print('{} broken results'.format(counter))
        pass

    save_figure('%s_zoom.pdf' % fname.replace('.pdf', ''), bbox_inches='tight', dpi=300)
    plt.close()

#####################
# Extractor functions
# value functions
def annihilate_low_correlations(vals, threshold=0.1):
    """ Take care of small fluctuations around 0
    """
    vals[abs(vals) <= threshold] = 0
    return vals

def bin_correlations(vals, low_thres=-0.1, high_thres=0.1):
    """ Bin `vals` into three categories
    """
    tmp = np.zeros(vals.shape)
    tmp[vals < low_thres] = -1
    tmp[vals > high_thres] = 1
    return tmp

def get_sign_changes(raw_vals, enh_vals):
    """ Compute number of sign changes
    """
    raw_vals = annihilate_low_correlations(raw_vals)
    enh_vals = annihilate_low_correlations(enh_vals)
    return np.sum(np.invert(np.sign(raw_vals) == np.sign(enh_vals)))

def get_rank_changes(raw_vals, enh_vals):
    """ Detect changes in the order of correlations
    """
    raw_vals = bin_correlations(raw_vals)
    enh_vals = bin_correlations(enh_vals)
    return np.sum(np.invert(np.argsort(raw_vals) == np.argsort(enh_vals)))

# sorting functions
def sort_by_network_density(netw):
    """network density"""
    edge_num = np.count_nonzero(netw.jacobian)
    max_edge_num = netw.jacobian.shape[0]**2
    return round(edge_num / max_edge_num, 2)

def sort_by_indeg(netw):
    """in-degree of 'last' node"""
    in_vec = netw.jacobian[:,-1][:-1]
    return np.sum(in_vec)

def sort_by_outdeg(netw):
    """out-degree of 'last' node"""
    out_vec = netw.jacobian[-1,:][:-1]
    return np.sum(out_vec)

def sort_by_cycle_num(netw):
    """number of cycles"""
    graph = nx.from_numpy_matrix(netw.jacobian, create_using=nx.DiGraph())
    return len(list(nx.simple_cycles(graph)))

def handle_plots(inp):
    """ Generate plots for varying data extraction functions
    """
    for vfunc, title in zip([get_sign_changes, get_rank_changes], ['sign', 'rank']):
        ptitle = '{} changes'.format(title)

        # clustering
        for clus_typ in ['hamming', 'minkowski']:
            plot_result(inp,
                vfunc, (sort_by_outdeg, 'cluster:{}'.format(clus_typ)),
                ptitle, 'images/matrix_{}_outdeg_{}.pdf'.format(title, clus_typ))
            plot_result(inp,
                vfunc, (sort_by_indeg, 'cluster:{}'.format(clus_typ)),
                ptitle, 'images/matrix_{}_indeg_{}.pdf'.format(title, clus_typ))
            plot_result(inp,
                vfunc, (sort_by_network_density, 'cluster:{}'.format(clus_typ)),
                ptitle, 'images/matrix_{}_netdens_{}.pdf'.format(title, clus_typ))
            plot_result(inp,
                vfunc, (sort_by_cycle_num, 'cluster:{}'.format(clus_typ)),
                ptitle, 'images/matrix_{}_cycles_{}.pdf'.format(title, clus_typ))

        # vanilla matrices
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
    if len(sys.argv) == 1:
        fname = 'results/matrix_data.dat'
        generate_data(fname)
    elif len(sys.argv) == 2:
        fname = sys.argv[1]
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
        handle_plots(inp)
    else:
        print('Usage: %s [data file]' % sys.argv[0])
        sys.exit(-1)

if __name__ == '__main__':
    main()
