"""
Parse real-life peak data
"""

import re
import sys
import csv
import itertools
import collections

import numpy as np
import networkx as nx
import scipy.stats as scits
import matplotlib.pylab as plt

from tqdm import tqdm


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
    return data

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

    print('Found %d networks in %d educts and %d products' \
        % (len(motifs), len(educts), len(products)))
    return motifs

def get_complete_network(data, strict=True):
    """ Generate complete network hidden in data.
        In `strict` mode, products are only considered if both educts exist as well
    """
    graph = nx.DiGraph()

    # create graph
    if strict:
        motifs = find_3_node_networks(data)
        for entry in motifs:
            e1, t, e2 = entry[0]

            graph.add_node(t, color='blue')
            graph.add_edge(e1, t, color='gray')
            graph.add_edge(e2, t, color='gray')
    else:
        for typ, spec in data.keys():
            if typ == 'product':
                graph.add_node(spec[1], color='blue')
                graph.add_edge(spec[0], spec[1], color='gray')
                graph.add_edge(spec[2], spec[1], color='gray')
            elif typ == 'educt':
                graph.add_node(spec)
            else:
                raise RuntimeError('Unknown entry "%s"' % str((typ, spec)))

    # list some information
    lwcc = max(nx.weakly_connected_component_subgraphs(graph), key=len)
    print('Largest weakly connected component:', len(lwcc))

    # plot graph
    pydot_graph = nx.nx_pydot.to_pydot(graph)
    pydot_graph.write_pdf(
        'images/complete_peak_network.pdf',
        prog=['fdp', '-Goutputorder=edgesfirst'])

def compute_correlations(motifs):
    """ Compute correlation histograms for all intensity-list pairs
    """
    def do_hist(name1, ints1, name2, ints2):
        corrs = []
        for i1, i2 in itertools.product(ints1, ints2):
            corr, p = scits.pearsonr(i1, i2)
            corrs.append(corr)

        bin_edges = np.linspace(-1, 1, 200)
        n, _, _ = plt.hist(
            corrs, bin_edges, facecolor='khaki')

        plt.title('"{}" vs "{}"'.format(name2, name1))
        plt.xlabel('correlation')
        plt.ylabel('count')

        plt.savefig('corr_hists/corr_hist_{}_{}.pdf'.format(name1, name2))

    for spec, ints in tqdm(motifs):
        e1_ints, p_ints, e2_ints = ints
        p_name = '{} [{}] {}'.format(*spec)

        do_hist(spec[0], e1_ints, spec[2], e2_ints)
        do_hist(p_name, p_ints, spec[0], e1_ints)
        do_hist(p_name, p_ints, spec[2], e2_ints)


def main(fname):
    """ Analyse peaks
    """
    data = read_file(fname)

    get_complete_network(data)

    res = find_3_node_networks(data)
    compute_correlations(res)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <peak file>' % sys.argv[0])
        sys.exit(-1)

    main(sys.argv[1])
