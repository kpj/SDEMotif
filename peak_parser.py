"""
Parse real-life peak data
"""

import re
import sys
import csv
import collections

import networkx as nx


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
                (mac,) if res is None else res.groups()
            ))
        return out

    def parse_intensities(ints):
        """ Filter zero-entries
        """
        return map(float, ints)

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
                data[entry_pair].extend(
                    parse_intensities(row[int_slice]))
    return data

def find_3_node_networks(data):
    """ Find two-substrate reactions and product
    """
    def get_intensities(typ, crit):
        """ Extract macthing intensities
        """
        ints = []
        for (t, rest), series in data.items():
            if t == typ and rest == crit:
                ints.extend(series)
        return ints

    educts = [com[1] for com in data.keys() if com[0] == 'educt']
    products = [com[1] for com in data.keys() if com[0] == 'product']

    # find all motifs
    nets = []
    for e1, t, e2 in products:
        if e1 in educts and e2 in educts:
            prod = '%s, %s, %s' % (e1, t, e2)
            nets.append((
                (e1, e2, prod),
                (get_intensities('educt', e1), get_intensities('educt', e2), get_intensities('product', (e1, t, e2)))
            ))

    print('Found %d networks in %d educts and %d products' \
        % (len(nets), len(educts), len(products)))

    # select motifs with enough intensities
    motifs = []
    for ex in nets:
        ls = [len(l) for l in ex[1]]
        if len(set(ls)) == 1:
            motifs.append(ex)

    print('%d of them are suitable' % len(motifs))
    for m in motifs:
        print(m[0])

    return motifs

def get_complete_network(data):
    """ Generate complete network hidden in data
    """
    graph = nx.DiGraph()

    # create graph
    for typ, spec in data.keys():
        if typ == 'product':
            graph.add_node(spec[1], color='blue')
            graph.add_edge(spec[0], spec[1], color='gray')
            graph.add_edge(spec[2], spec[1], color='gray')
        elif typ == 'educt':
            graph.add_node(spec[0])
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

def main(fname):
    """ Analyse peaks
    """
    data = read_file(fname)
    res = find_3_node_networks(data)
    get_complete_network(data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <peak file>' % sys.argv[0])
        sys.exit(-1)

    main(sys.argv[1])
