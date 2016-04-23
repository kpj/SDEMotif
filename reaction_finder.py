"""
Find reactions via combinatoric investigations of data files
"""

import csv
import itertools
import collections

import numpy as np
import pandas as pd
import scipy.stats as scis

import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns

from tqdm import tqdm

import plotter


ATOM_LIST = 'CHONS'

def read_compounds_file(file_spec):
    """ Transform data from compounds file into usable format:

        {
            <compound name>: {
                'groups': {
                    <chemical group>: <amount>,
                    ...
                },
                'mass': <mass>
            },
            ...
        }
    """
    data = {}

    with open(file_spec, 'r') if isinstance(file_spec, str) else file_spec as fd:
        reader = csv.reader(fd)

        # parse header
        head = next(reader)

        group_cols, group_names = zip(*[p for p in enumerate(head)
            if p[1].startswith('-')])
        group_range = range(group_cols[0], group_cols[-1]+1)

        atom_range = {}
        for a in ATOM_LIST:
            try:
                atom_range[head.index(a)] = a
            except ValueError:
                pass

        name_ind = head.index('Name')
        mass_ind = head.index('M-H')

        # parse body
        for row in reader:
            name = row[name_ind].strip()
            mass = row[mass_ind]
            if mass == 'NA': continue

            data[name] = {
                'groups': {},
                'atoms': {},
                'mass': float(mass)
            }
            for ind in group_range:
                data[name]['groups'][group_names[ind-group_cols[0]]] = int(row[ind])

            for idx, atom in atom_range.items():
                data[name]['atoms'][atom] = int(row[idx])

    return data

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
                'group_trans': {
                    <chemical group>: <amount>,
                    ...
                },
                'mass_trans': <mass transformation>
            },
            ...
        }
    """
    data = {}

    with open(file_spec, 'r') if isinstance(file_spec, str) else file_spec as fd:
        reader = csv.reader(fd)

        head = next(reader)
        groups = next(reader)

        rname_ind = head.index('Reaction')
        rmass_ind = head.index('Mass Addendum')
        trans_ind = head.index('Transformation')

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
            atom_trans = parse_atom_transformation(row[trans_ind])

            data[name] = {
                'mass_trans': float(mass),
                'atom_trans': atom_trans,
                'c1': {},
                'c2': {},
                'group_trans': {}
            }

            for i in c1_reqs_ra:
                data[name]['c1'][groups[i-c1_reqs_ra[0]+1]] = int(row[i])
            for i in c2_reqs_ra:
                try:
                    data[name]['c2'][groups[i-c2_reqs_ra[0]+1]] = int(row[i])
                except ValueError:
                    data[name]['c2'] = None
                    break
            for i in res_mod_ra:
                data[name]['group_trans'][groups[i-res_mod_ra[0]+1]] = int(row[i])

    return data

def parse_atom_transformation(string):
    """ Parse reaction atom transformation description
    """
    cur_sign = 1
    c1, c2 = False, False
    data = {}

    state = 'idle'
    cur_atom = None

    for char in string:
        if char == '+':
            cur_sign = 1
        elif char == '-':
            cur_sign = -1
        else:
            if state == 'idle':
                if char == 'M':
                    state = 'parseM'
                elif char == ' ':
                    continue
                elif char in ATOM_LIST:
                    state = 'parseA'
                    cur_atom = char
                else:
                    raise RuntimeError('Encountered unexpected character: "{}" while in state "{}"'.format(char, state))
            elif state == 'parseM':
                if char == '1':
                    c1 = True
                elif char == '2':
                    c2 = True
                else:
                    state = 'idle'
            elif state == 'parseA':
                if char.isdigit():
                    data[cur_atom] = cur_sign * int(char)
                    cur_atom = None
                elif char == ' ':
                    data[cur_atom] = cur_sign * 1
                    cur_atom = None
                    state = 'idle'
                else:
                    if cur_atom is None:
                        cur_atom = char
                    data[cur_atom] = cur_sign * 1
    if state == 'parseA' and not cur_atom is None:
        data[cur_atom] = cur_sign * 1

    data.update({
        'c1': c1,
        'c2': c2
    })

    return data

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
        if (
            c2 is None and
            spec['c2'] is None and
            match(cdata[c1]['groups'], spec['c1'])
        ) or (
            not c2 is None and
            match(cdata[c1]['groups'], spec['c1']) and
            match(cdata[c2]['groups'], spec['c2'])
        ):
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
    """ Infer new compounds from reactions of existing ones.

        All kinds of new information computation take place here
    """
    def add_specs(*args):
        spec = {}
        for k in args[0]:
            spec[k] = sum([s[k] for s in args if k in s])
        return spec

    data = {}
    for rname, pairs in combs.items():
        r_groups = rdata[rname]['group_trans']
        r_mass = rdata[rname]['mass_trans']
        r_atoms = rdata[rname]['atom_trans']

        for c1, c2 in pairs:
            # compute new groups
            c1_groups = cdata[c1]['groups']
            c2_groups = cdata[c2]['groups'] if not c2 is None else {}

            new_groups = add_specs(c1_groups, c2_groups, r_groups)

            for g in list(c1_groups.keys()) + list(c2_groups.keys()):
                if not g in new_groups: new_groups[g] = 0

            # compute new result
            c1_mass = cdata[c1]['mass']
            c2_mass = cdata[c2]['mass'] if not c2 is None else 0

            new_mass = c1_mass + r_mass + c2_mass

            # compute new atoms
            c1_atoms = cdata[c1]['atoms'] if r_atoms['c1'] else {}
            c2_atoms = cdata[c2]['atoms'] if r_atoms['c2'] else {}
            r_trans = {k: v for k,v in r_atoms.items() if k in ATOM_LIST}

            new_atoms = add_specs(c1_atoms, c2_atoms, r_trans)

            # store results
            new_name = '({c1}) {{{r}}} ({c2})'.format(r=rname, c1=c1, c2=c2)

            data[new_name] = {
                'groups': new_groups,
                'mass': new_mass,
                'atoms': new_atoms
            }

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

def read_peak_data(fname):
    """ Parse peak data file
    """
    data = {}
    with open(fname) as fd:
        reader = csv.DictReader(fd)
        sample_keys = [k for k in reader.fieldnames if k.startswith('LC.MS.')]

        for row in reader:
            mass = row['mz']
            ints = [float(row[k]) for k in sample_keys]

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
    for name, dic in masses.items():
        res = match(dic['mass'])

        if len(res) > 0:
            data[name] = res

    return data

def get_correlation_matrix(sols):
    """ Compute correlation matrix for given list of time series
    """
    dim = len(sols)
    mat = np.empty((dim, dim))

    for i in range(dim):
        for j in range(dim):
            cc, _ = scis.pearsonr(sols[i], sols[j])
            mat[i, j] = cc

    return mat

def plot_all_correlations(comps, intensities, ax):
    """ Plot correlations between all intensity combinations
    """
    col_list = ['red', 'blue', 'green']

    tmp = collections.defaultdict(lambda: collections.defaultdict(list))
    for c1 in comps:
        for c2 in comps:
            if c1 == c2: break
            corrs = {}

            # compute correlations
            for i, int1 in enumerate(intensities[c1]):
                for j, int2 in enumerate(intensities[c2]):
                    cc, _ = scis.pearsonr(int1, int2)
                    corrs[(i,j)] = cc

            c1_idx, c2_idx = max(corrs.keys(), key=lambda k: corrs[k])
            tmp[c1][c1_idx].append(corrs[(c1_idx, c2_idx)])
            tmp[c2][c2_idx].append(corrs[(c1_idx, c2_idx)])

            # plot histogram
            plotter.plot_histogram(
                list(corrs.values()), ax,
                alpha=0.5, facecolor=col_list.pop())

    # choose final selection
    sel = {}
    for c, maps in tmp.items():
        choice = max(maps.keys(), key=lambda k: sum(maps[k]))
        sel[c] = choice

    return sel

def plot_result(motifs):
    """ Create result plot
    """
    mpl.style.use('default')

    plt.figure(figsize=(30, 4 * len(motifs)))
    gs = mpl.gridspec.GridSpec(len(motifs), 3, width_ratios=[1, 2, 1])

    for i, (c1, c2, c3, intensities) in enumerate(motifs):
        # plot all possible correlations and select optimal one
        sel = plot_all_correlations([c1, c2, c3], intensities, plt.subplot(gs[i, 2]))

        # get intensities
        sols = []
        for foo in [c1, c2, c3]:
            sols.append(intensities[foo][sel[foo]])

        # compute correlation matrix
        corr_mat = get_correlation_matrix(sols)

        # plot rest
        plotter.plot_corr_mat(corr_mat, plt.subplot(gs[i, 0]))

        series_ax = plt.subplot(gs[i, 1])
        plotter.plot_system_evolution(
            sols, series_ax,
            xlabel='sample')
        series_ax.set_title(c3)

    plt.tight_layout()
    plotter.save_figure('images/rl_motifs.pdf', bbox_inches='tight')
    plt.close()

def find_small_motifs(
    compounds_level0, intensities_level0,
    reaction_data,
    num=10
):
    """ Look for 3 node motifs
    """
    comp_tmp = iterate_once(compounds_level0, reaction_data)

    intensities_level1 = match_masses(comp_tmp)
    compounds_level1 = {k: comp_tmp[k] for k in intensities_level1.keys()}

    print('Found {} new compounds'.format(len(compounds_level1)))

    # find 3 motif networks
    motifs = []
    used_compounds = set()
    done = False
    with tqdm(total=num) as pbar:
        for comp_level0, groups_level0 in compounds_level0.items():
            for comp_level1, groups_level1 in compounds_level1.items():
                c1_level1, _, c2_level1 = parse_compound_name(comp_level1)
                if not comp_level0 in [c1_level1, c2_level1]: continue

                # combine old with new components
                comp_tmp = iterate_once(
                    dict([
                        (comp_level0, groups_level0),
                        (comp_level1, groups_level1)
                    ]), reaction_data)

                # check which are associated with intensities
                intensities_level2 = match_masses(comp_tmp)
                compounds_level2 = {k: comp_tmp[k] for k in intensities_level2.keys()}

                # find actual 3 node motifs
                for comp_level2, groups_level2 in compounds_level2.items():
                    c1_level2, _, c2_level2 = parse_compound_name(comp_level2)

                    # connected motif check
                    if not ((comp_level0 == c1_level2 and comp_level1 == c2_level2) or (comp_level0 == c2_level2 and comp_level1 == c1_level2)):
                        continue

                    # compounds not already used
                    if comp_level0 in used_compounds or comp_level1 in used_compounds or comp_level2 in used_compounds:
                        continue

                    # compute all intensity vectors
                    intensities_all = {}
                    intensities_all.update(intensities_level0)
                    intensities_all.update(intensities_level1)
                    intensities_all.update(intensities_level2)

                    # save result
                    motifs.append(
                        (comp_level0, comp_level1, comp_level2, intensities_all))

                    used_compounds.update([comp_level0, comp_level1, comp_level2])

                    pbar.update()
                    if len(motifs) >= num:
                        done = True

                    if done: break
                if done: break
            if done: break

    # plot stuff
    plot_result(motifs)

def investigate_reactions(
    compounds_level0, intensities_level0,
    reaction_data
):
    """ Find out if reactions induce correlation patterns
    """
    # combine initial compounds
    comp_tmp = iterate_once(compounds_level0, reaction_data)

    intensities_level1 = match_masses(comp_tmp)
    print('Found {} new compounds'.format(len(intensities_level1)))

    intensities_all = {}
    intensities_all.update(intensities_level0)
    intensities_all.update(intensities_level1)

    # compute all correlations for all compounds
    rea_corrs = []
    for compound in tqdm(intensities_level1.keys()):
        c1, rea, c2 = parse_compound_name(compound)
        if c2 == 'None': continue

        for int1 in intensities_all[c1]:
            for int2 in intensities_all[c2]:
                cc, _ = scis.pearsonr(int1, int2)
                rea_corrs.append({'reaction': rea, 'correlation': cc})
    df = pd.DataFrame.from_dict(rea_corrs)

    # plot result
    fig = plt.figure()
    for rea in df.reaction.unique():
        corrs = df[df.reaction==rea].correlation
        sns.distplot(
            corrs, label=rea,
            kde=False, bins=np.linspace(-1, 1, 200))

    plt.legend(loc='best')

    plt.tight_layout()
    plotter.save_figure('images/rl_reaction_patterns.pdf', bbox_inches='tight')
    plt.close()

def list_structure_formulas(
    compounds_level0, intensities_level0,
    reaction_data
):
    """ Investigate structural formulas of molecules
    """
    # generate data
    comp_tmp = iterate_once(compounds_level0, reaction_data)

    intensities_level1 = match_masses(comp_tmp)
    compounds_level1 = {k: comp_tmp[k] for k in intensities_level1.keys()}

    # print data
    for name, data in compounds_level0.items():
        print(name, data['atoms'])

    for name, data in compounds_level1.items():
        print(name, data['atoms'])

def main(compound_fname, reaction_fname):
    """ Read in data and start experiment
    """
    compound_data = read_compounds_file(compound_fname)
    reaction_data = read_reactions_file(reaction_fname)

    # only keep masses which have associated intensities
    intensities_level0 = match_masses(compound_data)
    compounds_level0 = {k: compound_data[k] for k in intensities_level0.keys()}

    # find new compounds
    print('Starting with {} compounds'.format(len(compounds_level0)))

    # investigate results
    list_structure_formulas(
        compounds_level0, intensities_level0,
        reaction_data)

    investigate_reactions(
        compounds_level0, intensities_level0,
        reaction_data)

    find_small_motifs(
        compounds_level0, intensities_level0,
        reaction_data)


if __name__ == '__main__':
    main('data/Compound_List.csv', 'data/Reaction_List.csv')
