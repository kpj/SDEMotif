"""
Parse real-life peak data
"""

import re
import sys
import csv
import json
import itertools
import collections

import numpy as np
import networkx as nx
import scipy.stats as scits
import matplotlib.pylab as plt
import matplotlib as mpl

from tqdm import tqdm

from system import SDESystem
from main import analyze_system
from plotter import save_figure, plot_corr_mat, plot_system_evolution, plot_histogram
from utils import extract_sub_matrix, list_diff


def read_file(fname):
    """ Read file into some data structure:
        {
            'compound name': [<int1>, <int2>, ...],
            ...
        }
    """
    def parse_compound_name(name):
        """ Return individual compounds in given name
        """
        out = []
        macros = re.findall(r'\[(.*?)\]', name)
        for mac in macros:
            res = re.match(r'^\((.*?)\) (.*) \((.*?)\)$', mac)
            out.append((
                'educt' if res is None else 'product',
                mac if res is None else res.groups()
            ))
        return out

    def parse_intensities(ints):
        """ Filter zero-entries
        """
        return list(map(float, ints))

    data = collections.defaultdict(list)
    with open(fname, 'r') as fd:
        reader = csv.reader(fd)

        # parse header
        intensity_cols = [i for i, name in enumerate(next(reader))
            if name.startswith('LC.MS')]
        int_slice = slice(intensity_cols[0], intensity_cols[-1]+1)

        # parse data
        for row in reader:
            cname = row[-1]
            for entry_pair in parse_compound_name(cname):
                data[entry_pair].append(
                    parse_intensities(row[int_slice]))
    return dict(data)

def find_3_node_networks(data):
    """ Find two-substrate reactions and product
    """
    def get_intensities(typ, crit):
        """ Extract matching intensities
        """
        return data[(typ, crit)]

    educts = [com[1] for com in data.keys() if com[0] == 'educt']
    products = [com[1] for com in data.keys() if com[0] == 'product']

    # find all motifs
    motifs = []
    for e1, t, e2 in products:
        if e1 in educts and e2 in educts:
            motifs.append((
                (e1, t, e2),
                (get_intensities('educt', e1), get_intensities('product', (e1, t, e2)), get_intensities('educt', e2))
            ))

    print('>', 'Found %d networks in %d educts and %d products' \
        % (len(motifs), len(educts), len(products)))
    return motifs

def get_complete_network(data, strict=True, plot=True):
    """ Generate complete network hidden in data.
        In `strict` mode, products are only considered if both educts exist as well
    """
    graph = nx.DiGraph()

    # create graph
    if strict:
        motifs = find_3_node_networks(data)
        for entry in motifs:
            e1, t, e2 = entry[0]
            node_label = '{}, {}, {}'.format(*entry[0])

            graph.add_node(node_label, color='blue')
            graph.add_edge(e1, node_label, color='gray')
            graph.add_edge(e2, node_label, color='gray')
    else:
        for typ, spec in data.keys():
            if typ == 'product':
                node_label = '{}, {}, {}'.format(*spec)
                graph.add_node(node_label, color='blue')
                graph.add_edge(spec[0], node_label, color='gray')
                graph.add_edge(spec[2], node_label, color='gray')
            elif typ == 'educt':
                graph.add_node(spec)
            else:
                raise RuntimeError('Unknown entry "%s"' % str((typ, spec)))

    # list some information
    lwcc = max(nx.weakly_connected_component_subgraphs(graph), key=len)
    print('>', 'Largest weakly connected component:', len(lwcc))

    # plot graph
    if plot:
        pydot_graph = nx.nx_pydot.to_pydot(graph)
        pydot_graph.write_pdf(
            'images/complete_peak_network.pdf',
            prog=['fdp', '-Goutputorder=edgesfirst'])

    return graph

def simulate_graph(graph):
    """ Generate dynamics on graph
    """
    # create system
    J = np.copy(nx.to_numpy_matrix(graph))
    np.fill_diagonal(J, -1)
    D = np.zeros((J.shape[0],))
    E = np.zeros((J.shape[0],))
    I = np.ones((J.shape[0],))

    # add input to nodes of zero in-degree
    zero_indgr = []
    for i, row in enumerate(J.T):
        inp = np.sum(row)
        inp += 1 # compensate for self-inhibition
        if inp == 0: zero_indgr.append(i)

    D[zero_indgr] = 1
    E[zero_indgr] = 1
    print('>', '{}/{} nodes with zero indegree'.format(len(zero_indgr), len(graph.nodes())))

    # simulate system
    syst = SDESystem(J, D, E, I)
    syst, mat, sol = analyze_system(syst, filter_trivial_ss=False)

    # plot results
    fig = plt.figure(figsize=(30, 15))
    gs = mpl.gridspec.GridSpec(1, 2, width_ratios=[1, 2])

    if not mat is None:
        # only keep non-zero indegree node correlations
        mat = extract_sub_matrix(mat, zero_indgr)

        node_inds = list_diff(range(J.shape[0]), zero_indgr)
        used_nodes = np.array(graph.nodes())[node_inds]

        plot_corr_mat(
            mat, plt.subplot(gs[0]),
            show_values=False, labels=used_nodes)
    plot_system_evolution(sol, plt.subplot(gs[1]), show_legend=False)

    save_figure('images/peak_network_simulation.pdf', bbox_inches='tight', dpi=300)

def compute_correlation_pairs(motifs, plot=False):
    """ Compute correlation histograms for all intensity-list pairs
    """
    def do_hist(name1, ints1, name2, ints2):
        corrs = []
        for i1, i2 in itertools.product(ints1, ints2):
            corr, p = scits.pearsonr(i1, i2)
            corrs.append(corr)

        if plot:
            plt.figure()
            plot_histogram(corrs, plt.gca())

            plt.title('"{} ({})" vs "{} ({})"'.format(
                name2, len(ints2), name1, len(ints1)))
            plt.xlabel('correlation')

            plt.savefig('corr_hists/corr_hist_{}_{}.pdf'.format(name1, name2))
            plt.close()

        return corrs

    res = []
    for spec, ints in tqdm(motifs):
        e1_ints, p_ints, e2_ints = ints
        p_name = '{} [{}] {}'.format(*spec)

        res.append(do_hist(spec[0], e1_ints, spec[2], e2_ints))
        res.append(do_hist(p_name, p_ints, spec[0], e1_ints))
        res.append(do_hist(p_name, p_ints, spec[2], e2_ints))
    return res

def compute_overview_histogram(corrs):
    """ Compute histogram of all possible correlations
    """
    flat_corrs = list(itertools.chain.from_iterable(corrs))

    plt.figure()
    plot_histogram(flat_corrs, plt.gca())

    plt.xlabel('molecule intensity correlation')
    plt.title('Overview over all correlations')

    plt.savefig('images/all_corrs_hist.pdf')
    plt.close()

    return flat_corrs

def intersect_flavour_handbook(data, mol_map_file):
    """ Check which molecules from the Handbook we have data for
    """
    with open(mol_map_file, 'r') as fd:
        mmap = json.load(fd)

    mols = set(list(mmap.keys()) + list([v for vl in mmap.values() for v in vl]))
    for typ, spec in data:
        if typ == 'educt':
            print(spec in mmap)


def main(fname):
    """ Analyse peaks
    """
    data = read_file(fname)

    intersect_flavour_handbook(data, 'data/synonym_map.json')

    graph = get_complete_network(data, plot=False)
    simulate_graph(graph)

    res = find_3_node_networks(data)
    all_corrs = compute_correlation_pairs(res, plot=False)
    compute_overview_histogram(all_corrs)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <peak file>' % sys.argv[0])
        sys.exit(-1)

    main(sys.argv[1])
