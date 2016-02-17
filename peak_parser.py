"""
Parse real-life peak data
"""

import re
import sys
import csv
import collections

import itertools
import matplotlib.pylab as plt


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
        return re.findall(r'\[(.*?)\]', name)

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
            for name in parse_compound_name(cname):
                data[name].extend(
                    parse_intensities(row[int_slice]))
    return data

def find_3_node_networks(data):
    """ Find two-substrate reactions and product
    """
    ints = list(itertools.chain(*data.values()))
    n, bins, patches = plt.hist(
        ints, 100, log=True,
        facecolor='khaki')

    plt.xlabel('intensity')
    plt.ylabel('count')
    plt.title('peak intensity distribution')

    plt.show()

def main(fname):
    """ Analyse peaks
    """
    data = read_file(fname)
    res = find_3_node_networks(data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <peak file>' % sys.argv[0])
        sys.exit(-1)

    main(sys.argv[1])
