"""
Find reactions via combinatoric investigations of data files
"""

import csv
import itertools
import collections


def read_compounds_file(file_spec):
    """ Transform data from compounds file into usable format:

        {
            <compound name>: {
                <chemical group>: <amount>,
                ...
            },
            ...
        }
    """
    data = collections.defaultdict(dict)
    with open(file_spec, 'r') if isinstance(file_spec, str) else file_spec as fd:
        reader = csv.reader(fd)

        # parse header
        head = next(reader)

        group_cols, group_names = zip(*[p for p in enumerate(head)
            if p[1].startswith('-')])
        group_range = range(group_cols[0], group_cols[-1]+1)

        name_ind = head.index('Name')

        # parse body
        for row in reader:
            name = row[name_ind]
            for ind in group_range:
                data[name][group_names[ind-group_cols[0]]] = int(row[ind])

    return dict(data)

def read_reactions_file(file_spec):
    """ Transform data from reactions file into usable format:

        {
            <reaction name>: {
                'c1': {
                    <chemical group>: <amount>,
                    ...
                },
                'c2': {
                    <chemical group>: <amount>,
                    ...
                },
                'res': {
                    <chemical group>: <amount>,
                    ...
                }
            },
            ...
        }
    """
    data = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(file_spec, 'r') if isinstance(file_spec, str) else file_spec as fd:
        reader = csv.reader(fd)

        head = next(reader)
        groups = next(reader)

        rname_ind = head.index('Reaction')
        c1_reqs_ra = range(
            head.index('Requirement Matrix - Compound 1'),
            head.index('Requirement Matrix - Compound 2'))
        c2_reqs_ra = range(
            head.index('Requirement Matrix - Compound 2'),
            head.index('Result Matrix'))
        res_mod_ra = range(
            head.index('Result Matrix'),
            head.index('Transformation'))

        for row in reader:
            name = row[rname_ind]

            for i in c1_reqs_ra:
                data[name]['c1'][groups[i-c1_reqs_ra[0]+1]] = int(row[i])
            for i in c2_reqs_ra:
                try:
                    data[name]['c2'][groups[i-c2_reqs_ra[0]+1]] = int(row[i])
                except ValueError:
                    data[name]['c2'] = None
                    break
            for i in res_mod_ra:
                data[name]['res'][groups[i-res_mod_ra[0]+1]] = int(row[i])

    return dict(data)

def match(cgroups, react_groups):
    """ Check if given compound could be reaction partner at given `pos`
    """
    for group, amount in cgroups.items():
        if react_groups is None or amount < react_groups[group]:
            return False
    return True

def check_pair(c1, c2, cdata, rdata):
    """ Check whether given pair of compounds could react together
    """
    reacts = []
    for rname, spec in rdata.items():
        if (c2 is None and spec['c2'] is None and match(cdata[c1], spec['c1'])) or (not c2 is None and match(cdata[c1], spec['c1']) and match(cdata[c2], spec['c2'])):
            reacts.append(rname)
    return reacts

def combine_data(cdata, rdata):
    """ Combine compound and reaction data and extrapolate
    """
    # compute cross-product of all compounds
    perms = list(itertools.permutations(cdata.keys(), 2))
    data = collections.defaultdict(list)

    # find single reactions
    for c1 in cdata.keys():
        res = check_pair(c1, None, cdata, rdata)
        for react in res:
            data[react].append((c1, None))

    # find reaction partners
    for c1, c2 in perms:
        res = check_pair(c1, c2, cdata, rdata)
        for react in res:
            data[react].append((c1, c2))

    return data

def guess_new_compounds(combs, cdata, rdata):
    """ Infer new compounds from reactions of existing ones
    """
    m = lambda sp: collections.Counter(sp)

    data = {}
    for rname, pairs in combs.items():
        r_spec = rdata[rname]['res']
        for c1, c2 in pairs:
            c1_spec = cdata[c1]
            c2_spec = cdata[c2] if not c2 is None else {}

            new_spec = dict(m(c1_spec) + m(c2_spec) + m(r_spec))
            new_name = '({c1}) {r} ({c2})'.format(r=rname, c1=c1, c2=c2)

            for g in list(c1_spec.keys()) + list(c2_spec.keys()):
                if not g in new_spec: new_spec[g] = 0

            data[new_name] = new_spec

    return data

def iterate_once(compound_data, reaction_data):
    """ Find new products in given data
    """
    res = guess_new_compounds(
        combine_data(compound_data, reaction_data),
        compound_data, reaction_data)
    return res

def main(compound_fname, reaction_fname):
    """ Read in data and start experiment
    """
    compound_data = read_compounds_file(compound_fname)
    reaction_data = read_reactions_file(reaction_fname)

    res = iterate_once(compound_data, reaction_data)
    print('Found {} new compounds'.format(len(res)))

    # find further jumps
    for k, v in res.items():
        new_comps = {}
        new_comps.update(compound_data)
        new_comps.update({k: v})

        res2 = iterate_once(new_comps, reaction_data)

        # extract newly found jumps
        jumps = list(set.difference(set(res2), set(res)))
        print('Found {} new compounds [with {}]'.format(len(jumps), k))


if __name__ == '__main__':
    main('data/Compound_List.csv', 'data/Reaction_List.csv')
