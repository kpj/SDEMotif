"""
Investigate 3+1 node network with varied parameters
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm

from utils import extract_sig_entries
from plotter import save_figure, plot_system, plot_corr_mat, plot_system_evolution
from main import analyze_system
from setup import generate_basic_system


THRESHOLD = 0.2

def sort_columns(data, sort_data, sort_functions):
    """ Sort columns of `data` by multiple sort functions applied to `sort_data` in order
    """
    tmp = np.transpose(data).tolist()
    sort_tmp = np.copy(sort_data)
    inds = range(len(tmp))
    for sfunc in sort_functions[::-1]:
        tmp = [x for y, x in sorted(
            zip(sort_tmp, tmp), key=lambda pair: sfunc(pair[0][0]))]
        inds = [x for y, x in sorted(
            zip(sort_tmp, inds), key=lambda pair: sfunc(pair[0][0]))]
        sort_tmp = list(sorted(sort_tmp, key=lambda pair: sfunc(pair[0])))
    return np.transpose(tmp), inds

def handle_enh_entry(raw_res, enh_res, val_func):
    """ Compare given networks with given function
    """
    raw_sde, raw_odesde = raw_res
    enh_sde, enh_odesde = enh_res

    raw, raw_mat, raw_sol = raw_odesde
    enh, enh_mat, enh_sol = enh_odesde

    if raw_mat is None or enh_mat is None:
        return -1

    enh_mat = enh_mat[:-1,:-1] # disregard fourth node
    raw_vals = extract_sig_entries(raw_mat)
    enh_vals = extract_sig_entries(enh_mat)

    return val_func(raw_vals, enh_vals)

def preprocess_data(data, val_func, sort_functionality):
    """ Extract data information.
        Sort columns primarily by first sort_function and then the others in order
    """
    # compute matrix entries
    plot_data = []
    for raw, enh_res in data: # for each row
        plot_data.append([handle_enh_entry(raw, enh, val_func) for enh in enh_res])
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
    xtick_labels = np.array([xtick_func(n[0]) for n in char_netws])[repos]
    ytick_labels = [round(np.mean(abs(r[0][0].jacobian)), 2) for r, e in data]

    return plot_data, xtick_labels, ytick_labels

def cluster_data(mat, metric):
    """ Cluster given data
    """
    link = linkage(mat.T, metric=metric)

    dendr = dendrogram(
        link, no_plot=True)

    pos = dendr['leaves']
    return mat.T[pos].T, pos

def get_matrix_cmap():
    """ Assemble colormap for matrix
    """
    col_list = [(0.7,0.7,0.7), (0,0,1), (1,0,0), (0,1,0)]
    cmap = mpl.colors.ListedColormap(col_list, name='highlighter')
    cmap.set_under('white')
    return cmap

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

        plt.figure()
        cg = sns.clustermap(
            df, cmap=get_matrix_cmap(), vmin=0, vmax=3,
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

        plt.imshow(
            data,
            interpolation='nearest', cmap=get_matrix_cmap(),
            vmin=0, vmax=3)
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
        plot_individuals(netws_one, '{}_col_one'.format(fname), vfunc)
        plot_individuals(netws_two, '{}_col_two'.format(fname), vfunc)

        plot_individuals(netws_blue, '{}_row_blue'.format(fname))
        plot_individuals(netws_red, '{}_row_red'.format(fname))

    print('Done')

def select_column_by_jacobian(data, jac):
    """ Select column by approximated jacobian
    """
    ind = None
    for i, pair in enumerate(data[0][1]):
        syst, _, _ = pair[0]

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
            ref = data[i][0]
            net = data[i][1][ind]
            res.append((ref, net))

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

def plot_individuals(examples, fname, val_func=None):
    """ Plot a selection of individual results
    """
    if val_func is None:
        mod = -1
    else:
        mod = 0

    # plot selected networks
    if len(examples[0]) == 2: # is pair of networks
        fig = plt.figure(figsize=(50, 4*len(examples)))
        gs = mpl.gridspec.GridSpec(
            len(examples), 6+mod,
            width_ratios=[1, 2, 1, 2, 1+(-3*mod), 4])
    else: # each entry is single network
        fig = plt.figure(figsize=(25, 4*len(examples)))
        gs = mpl.gridspec.GridSpec(len(examples), 3, width_ratios=[1, 1, 2])

    counter = 0
    for i, net in enumerate(examples):
        if len(net) == 2: # pair of networks
            raw_p, enh_p = net

            # -.- ...
            if len(raw_p) == 2:
                _, raw = raw_p
                _, enh = enh_p
            else:
                raw = raw_p
                enh = enh_p

            plot_system(raw[0], plt.subplot(gs[i, 0]))
            plot_corr_mat(raw[1], plt.subplot(gs[i, 1]))
            plot_system(enh[0], plt.subplot(gs[i, 2]))
            plot_corr_mat(enh[1], plt.subplot(gs[i, 3]))
            plot_system_evolution(enh[2], plt.subplot(gs[i, 5+mod]))

            # plot marker
            mark_ax = plt.subplot(gs[i, 4])
            if not val_func is None:
                mark_ax.imshow(
                    [[handle_enh_entry(raw_p, enh_p, val_func)]],
                    cmap=get_matrix_cmap(), vmin=0, vmax=3)
                mark_ax.axis('off')
            else:
                print('Tried to use `val_func`, but it\'s None')
        else: # single network
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

    plt.tight_layout()
    save_figure('%s_zoom.pdf' % fname.replace('.pdf', ''), bbox_inches='tight', dpi=300)
    plt.close()

#####################
# Extractor functions
# value functions
def annihilate_low_correlations(vals, threshold=None):
    """ Take care of small fluctuations around 0
    """
    if threshold is None:
        threshold = THRESHOLD

    vals[abs(vals) < threshold] = 0
    return vals

def bin_correlations(vals, low_thres=None, high_thres=None):
    """ Bin `vals` into three categories
    """
    if low_thres is None:
        low_thres = -THRESHOLD
    if high_thres is None:
        high_thres = THRESHOLD

    tmp = np.zeros(vals.shape)
    tmp[vals < low_thres] = -1
    tmp[vals > high_thres] = 1
    return tmp

def get_sign_changes(raw_vals, enh_vals):
    """ Compute number of sign changes
    """
    raw_vals = annihilate_low_correlations(raw_vals)
    enh_vals = annihilate_low_correlations(enh_vals)

    nv_inds = np.intersect1d(np.nonzero(raw_vals), np.nonzero(enh_vals))
    nz_rw = raw_vals[nv_inds]
    nz_eh = enh_vals[nv_inds]

    return np.sum(np.invert(np.sign(nz_rw) == np.sign(nz_eh)))

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

def handle_input_spec(inp, spec):
    """ Only plot specified entries
        `spec` can be of the form:
        <value_func>|<sort_func>|<slice>,<slice>
        E.g.: 'get_sign_changes|sort_by_cycle_num|-1'
    """
    vfunc_str, sfunc_str, slices = spec.split('|')

    vfunc = globals()[vfunc_str]
    sfunc = globals()[sfunc_str]

    ex = lambda s: [int(e) if len(e) > 0 else None for e in s.split(':')]
    s1, s2 = slices.split(',')
    slc_row = slice(*ex(s1))
    slc_col = slice(*ex(s2))

    data, xticks, yticks = preprocess_data(inp['data'], vfunc, [sfunc])
    print(data[slc_row, slc_col])

def aggregate_motif_data(data, value_func=get_sign_changes, resolution=500):
    """ Compute sign-change frequency for range of thresholds
    """
    def find_threshold(data):
        """ Use std/2 of correlation distribution closest to 0 (most likely) to switch sign as detection threshold
        """
        cur = []
        for raw, enh_res in data:
            _, rd = raw
            _, rdm, _ = rd
            cur.append(extract_sig_entries(rdm))
            for enh in enh_res:
                _, ed = enh
                _, edm, _ = ed
                if not edm is None:
                    cur.append(extract_sig_entries(edm[:-1,:-1]))

        idx = np.argmin(abs(np.mean(cur, axis=0)))
        return np.std(cur, axis=0)[idx] / 2


    global THRESHOLD
    threshold_list = np.logspace(-5, 0, resolution)

    # produce data
    #first_data, last_data, std_data = None, None, None
    pairs = []
    for thres in tqdm(threshold_list):
        THRESHOLD = thres

        cur = []
        for raw, enh_res in data: # for each parameter configuration
            cur.append([handle_enh_entry(raw, enh, value_func) for enh in enh_res])
        cur = np.array(cur)

        #if thres == threshold_list[0]:
        #    first_data = cur
        #if thres == threshold_list[-1]:
        #    last_data = cur
        #if thres >= imp_thres and std_data is None:
        #    std_data = cur

        mat_res = np.sum(cur[cur>0])
        pairs.append((thres, mat_res))

    if cur.size == 0:
        return None, None, None

    print('Data shape:', cur.shape)
    total_num = cur[cur>=0].size * 3
    pairs = [(t,m/total_num) for t,m in pairs]

    # compute AUC of values right of threshold
    imp_thres = find_threshold(data)

    t_vals = [t for t,m in pairs if t >= imp_thres]
    m_vals = [m for t,m in pairs if t >= imp_thres]
    area = np.trapz(m_vals, x=t_vals)
    print('AUC:', area)

    return pairs, area, imp_thres

def threshold_influence(inp, value_func=get_sign_changes, resolution=500):
    """ Investigate influence of threshold
    """
    def plot_matrix(data):
        plt.tick_params(
            axis='both', which='both', labelleft='off',
            bottom='off', top='off', labelbottom='off', left='off', right='off')

        plt.imshow(
            data,
            interpolation='nearest', cmap=get_matrix_cmap(),
            vmin=0, vmax=3)
        plt.colorbar(ticks=range(np.max(data)+1), extend='min')

    # produce data
    pairs, area, imp_thres = aggregate_motif_data(
        np.asarray(inp['data']), value_func=value_func, resolution=resolution)

    # plot result
    value_func_name = value_func.__name__[4:]

    plt.figure()

    nz_vec = [(t, m) for t,m in pairs if m>0]
    z_vec = [(t, m) for t,m in pairs if m<=0]

    if len(nz_vec) > 0:
        plt.plot(*zip(*nz_vec), 'o')
    plt.plot(*zip(*z_vec), 'o', color='red')

    plt.axvspan(
        xmin=min([t for t,m in pairs]), xmax=imp_thres,
        alpha=0.1, color='blue')
    plt.annotate('half the correlation stdev ({:.02})'.format(imp_thres),
        xy=(imp_thres, .025), xycoords='data',
        xytext=(50, 20), textcoords='offset points',
        arrowprops=dict(arrowstyle='->'))

    plt.xscale('log')
    plt.title('Influence of binning threshold on number of {}'.format(value_func_name))
    plt.xlabel('binning threshold')
    plt.ylabel('frequency of {}'.format(value_func_name))

    # inside plots
    plt.style.use('default')

    #ax = plt.axes([0.1, 0.5, .2, .2])
    #plot_matrix(first_data)

    #ax = plt.axes([0.7, 0.4, .2, .2])
    #plot_matrix(last_data)

    #ax = plt.axes([0.4, 0.2, .2, .2])
    #plot_matrix(std_data)

    # save result
    save_figure('images/threshold_influence_{}.pdf'.format(value_func_name), bbox_inches='tight')

def plot_motif_overview(prefix):
    # get data
    data = {}
    pref_dir = os.path.dirname(prefix)
    for fn in os.listdir(pref_dir):
        if fn.startswith(os.path.basename(prefix)):
            fname = os.path.join(pref_dir, fn)
            print('>', fname)

            with open(fname, 'rb') as fd:
                inp = pickle.load(fd)

            motif = inp['data'][0][0][0][0] # *cough*
            _, area, _ = aggregate_motif_data(inp['data'], resolution=1)
            if not area is None:
                data[fn] = {
                    'idx': int(fn.split('_')[-1]),
                    'area': area,
                    'motif': motif
                }

    # plot data
    plt.figure()
    ax = plt.gca()

    motif_idx = []
    motif_rob = []
    for k in sorted(data):
        motif_idx.append(data[k]['idx'])
        motif_rob.append(data[k]['area'])

    plt.plot(motif_idx, motif_rob, 'o-')

    min_val, max_val = min(motif_idx)-1, max(motif_idx)+1
    plt.xlim(min_val, max_val)

    # add motif plots
    # TODO: finish this
    """axis_to_data = lambda x,y: ax.transData.inverted().transform(ax.transAxes.transform((x,y)))
    data_to_axis = lambda x,y: ax.transAxes.inverted().transform(ax.transData.transform((x,y)))
    axis_base, _ = data_to_axis(min_val+1, 0)

    for k in sorted(data):
        idx = int(k.split('_')[-1])
        x, _ = data_to_axis(idx, 0)
        print(idx, x)

        with plt.style.context(('default')):
            a = plt.axes([axis_base, 0.6, .3, .1])
            g = nx.from_numpy_matrix(data[k]['motif'].jacobian, create_using=nx.DiGraph())
            nx.draw(g, ax=a, node_size=60)
            a.axis('on')
            a.set_xticks([], [])
            a.set_yticks([], [])"""

    plt.savefig('images/motifs.pdf')

def main():
    """ Create matrix for various data functions
    """
    if len(sys.argv) == 2:
        fname = sys.argv[1]

        if os.path.exists(fname):
            with open(fname, 'rb') as fd:
                inp = pickle.load(fd)

            threshold_influence(inp)
            #threshold_influence(inp, value_func=get_rank_changes)

            #handle_plots(inp)
        else:
            # assume fname is motif prefix
            plot_motif_overview(fname)
    elif len(sys.argv) == 3:
        fname = sys.argv[1]
        with open(fname, 'rb') as fd:
            inp = pickle.load(fd)
        handle_input_spec(inp, sys.argv[2])
    else:
        print('Usage: %s [data file] [plot spec]' % sys.argv[0])
        sys.exit(-1)

if __name__ == '__main__':
    main()
