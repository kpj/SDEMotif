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
    group_data = collections.defaultdict(dict)
    mass_data = {}

    with open(file_spec, 'r') if isinstance(file_spec, str) else file_spec as fd:
        reader = csv.reader(fd)

        # parse header
        head = next(reader)

        group_cols, group_names = zip(*[p for p in enumerate(head)
            if p[1].startswith('-')])
        group_range = range(group_cols[0], group_cols[-1]+1)

        name_ind = head.index('Name')
        mass_ind = head.index('M-H')

        # parse body
        for row in reader:
            name = row[name_ind]
            mass = row[mass_ind]
            if mass == 'NA': continue

            for ind in group_range:
                group_data[name][group_names[ind-group_cols[0]]] = int(row[ind])
            mass_data[name] = float(mass)

    return dict(group_data), mass_data

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
    mdata = {}

    with open(file_spec, 'r') if isinstance(file_spec, str) else file_spec as fd:
        reader = csv.reader(fd)

        head = next(reader)
        groups = next(reader)

        rname_ind = head.index('Reaction')
        rmass_ind = head.index('Mass Addendum')

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
            mass = row[rmass_ind]

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

            mdata[name] = float(mass)

    return dict(data), mdata

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
    prod = list(itertools.product(cdata.keys(), repeat=2))
    data = collections.defaultdict(list)

    # find single reactions
    for c1 in cdata.keys():
        res = check_pair(c1, None, cdata, rdata)
        for react in res:
            data[react].append((c1, None))

    # find reaction partners
    for c1, c2 in prod:
        res = check_pair(c1, c2, cdata, rdata)
        for react in res:
            data[react].append((c1, c2))

    return data

def guess_new_compounds(combs, cdata, rdata):
    """ Infer new compounds from reactions of existing ones
    """
    def add_specs(*args):
        spec = {}
        for k in args[0]:
            spec[k] = sum([s[k] for s in args if k in s])
        return spec

    data = {}
    for rname, pairs in combs.items():
        r_spec = rdata[rname]['res']
        for c1, c2 in pairs:
            c1_spec = cdata[c1]
            c2_spec = cdata[c2] if not c2 is None else {}

            new_spec = add_specs(c1_spec, c2_spec, r_spec)
            new_name = '({c1}) {{{r}}} ({c2})'.format(r=rname, c1=c1, c2=c2)

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

def parse_compound_name(name):
    """ Parse name and return compounds and reaction
    """
    assert name[0] == '(' and name[-1] == ')'

    state = 'idle'
    depth = 0

    c1 = ''
    r = ''
    c2 = ''

    for char in name:
        if char in '({': depth += 1
        if char in ')}': depth -= 1

        if state == 'idle':
            if depth > 0:
                if len(c1) == 0: state = 'c1'
                elif len(r) == 0: state = 'r'
                elif len(c2) == 0: state = 'c2'
        elif state == 'c1':
            if depth > 0:
                c1 += char
            else:
                state = 'idle'
        elif state == 'r':
            if depth > 0:
                r += char
            else:
                state = 'idle'
        elif state == 'c2':
            if depth > 0:
                c2 += char
            else:
                state = 'idle'

    assert state == 'idle'
    return c1, r, c2

def compute_new_masses(new_compounds, comp_mass, rea_mass):
    """ Compute mass of new compounds from old ones plus reaction addendum
    """
    comp_mass['None'] = 0 # handle single compound cases

    mdata = {}
    for name in new_compounds.keys():
        c1, rea, c2 = parse_compound_name(name)

        new_mass = comp_mass[c1] + comp_mass[c2] + rea_mass[rea]
        mdata[name] = new_mass

    return mdata

def read_peak_data(fname):
    """ Parse peak data file
    """
    data = {}
    with open(fname) as fd:
        reader = csv.DictReader(fd)
        sample_keys = [k for k in reader.fieldnames if k.startswith('LC.MS.')]

        for row in reader:
            mass = row['mz']
            ints = [row[k] for k in sample_keys]

            data[float(mass)] = ints
    return data

def match_masses(masses):
    """ Match masses with entries from peak file
    """
    def match(mass, thres=1e-2):
        ms = []
        for km, ints in peak_data.items():
            if abs(km - mass) < thres:
                ms.append(ints)
        return ms

    peak_data = read_peak_data('data/peaklist_filtered_assigned.csv')

    data = {}
    for name, mass in masses.items():
        res = match(mass)

        if len(res) > 0:
            data[name] = res

    return data

def main(compound_fname, reaction_fname):
    """ Read in data and start experiment
    """
    compound_data, comp_mass = read_compounds_file(compound_fname)
    reaction_data, rea_mass = read_reactions_file(reaction_fname)

    # only keep masses which have associated intensities
    init_masses = match_masses(comp_mass)
    compound_data = {k: compound_data[k] for k in init_masses.keys()}

    print('Starting with {} compounds'.format(len(compound_data)))
    res = iterate_once(compound_data, reaction_data)
    print('Found {} new compounds'.format(len(res)))

    new_masses = compute_new_masses(res, comp_mass, rea_mass)
    mass_matches = match_masses(new_masses)
    print('Found {} mass matches'.format(len(mass_matches)))

    # find further jumps
    for name, intensities in mass_matches.items():
        # update data
        new_comps = {}
        new_comps.update(compound_data)
        new_comps.update({name: res[name]})

        all_masses = {}
        all_masses.update(comp_mass)
        all_masses.update({name: new_masses[name]})

        # next step
        res2 = iterate_once(new_comps, reaction_data)

        # extract newly found jumps
        jumps = list(set.difference(set(res2), set(res)))
        print(' > Found {} new compounds [with {}]'.format(len(jumps), name))

        # find intensities if needed
        if len(jumps) > 0:
            new_masses2 = compute_new_masses(res2, all_masses, rea_mass)
            mass_matches2 = match_masses(new_masses2)

            new_mass_matches = list(set.difference(set(mass_matches2), set(mass_matches)))
            print('  > Found {} new mass matches'.format(len(new_mass_matches)))

            if len(new_mass_matches) > 0:
                choice = new_mass_matches[0]
                print('  > {}'.format(choice))

                all_masses.update(new_masses2)
                all_mass_matches = {'None': 0}
                all_mass_matches.update(init_masses)
                all_mass_matches.update(mass_matches)
                all_mass_matches.update(mass_matches2)
                break

    # handle result
    print()
    print(choice)
    print(' >', all_masses[choice], len(all_mass_matches[choice]))

    c1, _, c2 = parse_compound_name(choice)
    print(c1)
    print(' >', all_masses[c1], len(all_mass_matches[c1]))
    print(c2)
    print(' >', all_masses[c2], len(all_mass_matches[c2]))

if __name__ == '__main__':
    main('data/Compound_List.csv', 'data/Reaction_List.csv')
