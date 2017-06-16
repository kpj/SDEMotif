"""
Find reactions via combinatoric investigations of data files
"""

import os
import csv
import copy
import pickle
import random
import itertools
import collections

import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats as scis

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes

from tqdm import tqdm, trange

import plotter
import utils
import formula_investigator


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
                'mass': float(mass),
                'origin': (None, None)
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

def gen_atom_string(atom_data):
    """ Convert atom-dict into readable string
    """
    out = ''
    for atom in ATOM_LIST:
        if atom in atom_data and atom_data[atom] != 0:
            out += '{}{}'.format(atom, atom_data[atom])
    return out

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
    for c1 in tqdm(cdata.keys()):
        res = check_pair(c1, None, cdata, rdata)
        for react in res:
            data[react].append((c1, None))

    # find reaction partners
    for c1, c2 in tqdm(prod):
        res = check_pair(c1, c2, cdata, rdata)
        for react in res:
            data[react].append((c1, c2))

    return dict(data)

def guess_new_compounds(combs, cdata, rdata):
    """ Infer new compounds from reactions of existing ones.

        All kinds of new information computations take place here
    """
    def add_specs(*args):
        spec = {}
        for k in args[0]:
            spec[k] = sum([s[k] for s in args if k in s])
        return spec

    data = {}
    for rname, pairs in tqdm(combs.items()):
        r_groups = rdata[rname]['group_trans']
        r_mass = rdata[rname]['mass_trans']
        r_atoms = rdata[rname]['atom_trans']

        for c1, c2 in tqdm(pairs):
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
                'atoms': new_atoms,
                'origin': (c1, c2)
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

def match_masses(masses, fname='data/peaklist_filtered_assigned.csv'):
    """ Match masses with entries from peak file
    """
    def match(mass, thres=1e-2):
        ms = []
        for km, ints in peak_data.items():
            if abs(km - mass) < thres:
                ms.append(ints)
        return ms

    peak_data = read_peak_data(fname)

    data = {}
    for name, dic in tqdm(masses.items()):
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

def plot_all_correlations(cs, data, ax):
    """ Plot correlations between all intensity combinations
    """
    col_list = ['red', 'blue', 'green']

    tmp = collections.defaultdict(lambda: collections.defaultdict(list))
    for c1 in cs:
        for c2 in cs:
            if c1 == c2: break
            corrs = {}

            # compute correlations
            for i, int1 in enumerate(data[c1]['intensities']):
                for j, int2 in enumerate(data[c2]['intensities']):
                    cc, _ = scis.pearsonr(int1, int2)
                    corrs[(i,j)] = cc

            c1_idx, c2_idx = max(corrs.keys(), key=lambda k: corrs[k])
            tmp[c1][c1_idx].append(corrs[(c1_idx, c2_idx)])
            tmp[c2][c2_idx].append(corrs[(c1_idx, c2_idx)])

            # plot histogram
            if not ax is None:
                plotter.plot_histogram(
                    list(corrs.values()), ax,
                    alpha=0.5, facecolor=col_list.pop())

    # choose final selection
    sel = {}
    for c, maps in tmp.items():
        choice = max(maps.keys(), key=lambda k: sum(maps[k]))
        sel[c] = choice

    return sel, corrs.values()

def plot_correlation_histogram(motifs, data):
    """ Plot histogram of all observed intensity correlations
    """
    colors = itertools.cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k'])

    plt.figure()
    for (m, lbl) in motifs:
        # compute correlations
        corrs = []
        for cs in tqdm(m):
            cur_corrs = []
            for c1 in cs:
                for c2 in cs:
                    if c1 == c2: break
                    for i, int1 in enumerate(data[c1]['intensities']):
                        for j, int2 in enumerate(data[c2]['intensities']):
                            cc, _ = scis.pearsonr(int1, int2)
                            cur_corrs.append(cc)
            #corrs.append(max(cur_corrs, key=abs))
            corrs.extend(cur_corrs)

        # plot
        sns.kdeplot(np.asarray(corrs), label=lbl)

    plt.title('Comparison of intensity correlation distributions')
    plt.xlabel('intensity vector correlation')
    plt.ylabel('frequency')

    plt.legend(loc='best')
    plt.tight_layout()
    plotter.save_figure('images/rl_corr_hist.pdf', bbox_inches='tight')

def plot_result(motifs, data, fname_app='', sub_num=10):
    """ Create result plot
    """
    mpl.style.use('default')
    sub_num = min(sub_num, len(motifs))
    print(' > Plotting results ({})'.format(fname_app[1:]))

    # overview plots
    plt.figure(figsize=(30, 4 * sub_num))
    gs = mpl.gridspec.GridSpec(sub_num, 3, width_ratios=[1, 2, 1])

    idx = map(int,
        np.linspace(0, len(motifs), num=sub_num, endpoint=False))

    corrs = []
    for ai, i in tqdm(enumerate(idx), total=sub_num):
        c1, c2, c3 = motifs[i]

        # plot all possible correlations and select optimal one
        sel, all_corrs = plot_all_correlations(
            (c1, c2, c3), data,
            plt.subplot(gs[ai, 2]) if ai < sub_num else None)

        # get intensities
        sols = []
        for foo in [c1, c2, c3]:
            sols.append(data[foo]['intensities'][sel[foo]])

        # compute correlation matrix
        corr_mat = get_correlation_matrix(sols)
        corrs.extend(all_corrs)

        # plot rest
        plotter.plot_corr_mat(corr_mat, plt.subplot(gs[ai, 0]))

        series_ax = plt.subplot(gs[ai, 1])
        plotter.plot_system_evolution(
            sols, series_ax,
            xlabel='sample')
        series_ax.set_title('{}\n{}\n{}'.format(c1, c2, c3))

    plt.tight_layout()
    plotter.save_figure('images/rl_motifs{}.pdf'.format(fname_app), bbox_inches='tight')

    # correlation histogram
    plt.figure()

    plotter.plot_histogram(corrs, plt.gca())

    plt.tight_layout()
    plotter.save_figure('images/rl_corr_hist{}.pdf'.format(fname_app), bbox_inches='tight')

def plot_network(motifs, data):
    """ Plot motif-network and show #intensity distribution
    """
    # generate graph
    motif_mem_counter = collections.defaultdict(int)
    graph = nx.DiGraph()
    for c1,c2,c3 in motifs:
        graph.add_edges_from([(c1,c2),(c1,c3),(c2,c3)])

        motif_mem_counter[c1] += 1
        motif_mem_counter[c2] += 1
        motif_mem_counter[c3] += 1
    motif_mem_counter = dict(motif_mem_counter)

    # generate overview
    overview = sorted(motif_mem_counter.keys(), key=lambda x: motif_mem_counter[x], reverse=True)
    legend_labels = ['Compound in motif occurence counter']
    for c in overview[:10]:
        cur = ' > {} {}'.format(c, motif_mem_counter[c])
        legend_labels.append(cur)

    # assignment counts
    ia_count = {n: len(data[n]['intensities']) for n in graph.nodes()}

    # plot graph
    plt.figure(figsize=(10, 12))

    pos = nx.nx_pydot.graphviz_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=[ia_count[n] for n in graph.nodes()])
    nx.draw_networkx_edges(graph, pos, alpha=.3, arrows=False, width=0.4)
    nx.draw_networkx_labels(graph, pos, labels=ia_count, font_size=1)

    plt.axis('on')
    plt.xticks([], [])
    plt.yticks([], [])

    plt.title('\n'.join(legend_labels), loc='left')
    plt.tight_layout()

    plt.savefig('images/motif_network.pdf')

def find_optimal_assignments(motifs, data, reps=1000, null_model=True, fname='motifs'):
    """ Find optimal compound assignments by (weighted) randomly selecting
        motifs of low initial assignment number and choose assignments
        which maximize intensity correlation coefficients.

        Goal:
            Enhance partially annotated MS peak file

        All steps:
            * Read compound/reaction data
                * note known MZ values
            * generate new compounds using reaction rules
                * compute theoretical MZ values
            * find motifs in all available compounds
                * assume that compounds in motifs have high intensity correlations

            * for each compound receive all intensity annotations from peak file
            * use randomized iterative procedure to find "optimal" MZ assignments
    """
    # find assignments
    def get_assignment_number(entry, assignments={}):
        c1, c2, c3 = entry

        num1 = len(data[c1]['intensities']) if c1 not in assignments else 1
        num2 = len(data[c2]['intensities']) if c2 not in assignments else 1
        num3 = len(data[c3]['intensities']) if c3 not in assignments and c3 is not None else 1

        return num1 * num2 * num3

    def filter_existing_intensities(entry, assignments):
        """ Remove intensity vectors from `cur['intensities']` which appear somewhere in `assignments`
        """
        entry = copy.deepcopy(entry)

        ints = entry['intensities']
        ass_ints = [d['intensities'][0] for c,d in assignments.items()]

        entry['intensities'] = [i for i in ints if i not in ass_ints]

        return entry

    def choose(entry, idx):
        """ Return entry with only `idx` intensity vector
        """
        entry = copy.deepcopy(entry)

        sel = entry['intensities'][idx]
        entry['intensities'] = [sel]

        return entry

    def assign(motifs, prob_fac=.2):
        assignments = {}

        sorted_motifs = sorted(
            motifs, key=get_assignment_number,
            reverse=True)
        sorted_motifs_bak = list(sorted_motifs)

        idx_list, cur_idx_list, assignment_order = [], [], []
        while len(sorted_motifs) > 0:
            # weighted choice of starting motif
            size = len(sorted_motifs)

            exp_vals = np.exp(prob_fac*np.arange(size))
            probs = exp_vals/np.sum(exp_vals)

            idx = np.random.choice(range(size), 1, p=probs)[0]
            entry = sorted_motifs[idx]
            sorted_motifs.remove(entry)

            idx_list.append(sorted_motifs_bak.index(entry))
            cur_idx_list.append(idx)

            c1, c2, c3 = entry
            cur = {
                c1: data[c1],
                c2: data[c2],
                c3: data[c3] if c3 is not None else {'intensities': []}
            }

            # process motif
            comps = c1, c2, c3

            # filter out already used intensity vectors
            fi = filter_existing_intensities

            if c3 is None: # dealing with link-like structure
                for c1 in comps:
                    for c2 in comps:
                        if c1 == c2: break
                        if c1 is None or c2 is None: break

                        corrs = {}

                        # skip if compounds are already assigned
                        if c1 in assignments and c2 in assignments:
                            continue

                        # compute correlations
                        c1_done, c2_done = c1 in assignments, c2 in assignments

                        data_1 = assignments[c1] if c1_done else fi(cur[c1], assignments)
                        data_2 = assignments[c2] if c2_done else fi(cur[c2], assignments)

                        if c1_done:
                            assert len(data_1['intensities']) == 1
                        if c2_done:
                            assert len(data_2['intensities']) == 1

                        for i, int1 in enumerate(data_1['intensities']):
                            for j, int2 in enumerate(data_2['intensities']):
                                if int1 == int2: continue
                                cc, _ = scis.pearsonr(int1, int2)
                                corrs[(i,j)] = cc

                        if len(corrs) == 0:
                            continue

                        # choose highest absolute correlation
                        c1_idx, c2_idx = max(corrs.keys(), key=lambda k: abs(corrs[k]))

                        if not c1_done:
                            assert not c1 in assignments
                            assignments[c1] = choose(data_1, c1_idx)
                            assignment_order.append(c1)
                        else:
                            assert c1_idx == 0
                        if not c2_done:
                            assert not c2 in assignments
                            assignments[c2] = choose(data_2, c2_idx)
                            assignment_order.append(c2)
                        else:
                            assert c2_idx == 0
            else: # dealing with 3-node motif
                if c1 in assignments and c2 in assignments and c3 in assignments:
                    continue

                # compute correlations
                c1_done, c2_done, c3_done = c1 in assignments, c2 in assignments, c3 in assignments

                data_1 = assignments[c1] if c1_done else fi(cur[c1], assignments)
                data_2 = assignments[c2] if c2_done else fi(cur[c2], assignments)
                data_3 = assignments[c3] if c3_done else fi(cur[c3], assignments)

                if c1_done:
                    assert len(data_1['intensities']) == 1
                if c2_done:
                    assert len(data_2['intensities']) == 1
                if c3_done:
                    assert len(data_3['intensities']) == 1

                corrs = {}
                for i, int1 in enumerate(data_1['intensities']):
                    for j, int2 in enumerate(data_2['intensities']):
                        for k, int3 in enumerate(data_3['intensities']):
                            if int1 == int2 or int2 == int3 or int3 == int1:
                                continue

                            cc1, _ = scis.pearsonr(int1, int2)
                            cc2, _ = scis.pearsonr(int2, int3)
                            cc3, _ = scis.pearsonr(int3, int1)
                            corrs[(i,j,k)] = abs(cc1) + abs(cc2) + abs(cc3)

                if len(corrs) == 0:
                    continue

                # choose highest absolute correlation
                c1_idx, c2_idx, c3_idx = max(corrs.keys(), key=lambda k: abs(corrs[k]))

                if not c1_done:
                    assert not c1 in assignments
                    assignments[c1] = choose(data_1, c1_idx)
                    assignment_order.append(c1)
                else:
                    assert c1_idx == 0
                if not c2_done:
                    assert not c2 in assignments
                    assignments[c2] = choose(data_2, c2_idx)
                    assignment_order.append(c2)
                else:
                    assert c2_idx == 0
                if not c3_done:
                    assert not c3 in assignments
                    assignments[c3] = choose(data_3, c3_idx)
                    assignment_order.append(c3)
                else:
                    assert c3_idx == 0

            # resort motifs
            sorted_motifs = sorted(
                sorted_motifs,
                key=lambda e: get_assignment_number(e, assignments),
                reverse=True)

        # check that all compounds are assigned to single, different intensity vectors
        all_ints = set()
        for c, cdata in assignments.items():
            assert len(cdata['intensities']) == 1, f'{c} got multiple assignments'
            all_ints.add(tuple(cdata['intensities'][0]))
        assert len(all_ints) == len(assignments), 'Some compounds are assigned to same intensity vector'

        extra = {
            'idx_list': idx_list,
            'cur_idx_list': cur_idx_list,
            'assignment_order': assignment_order
        }

        return assignments, extra

    def plot_correlations(assignments, ax):
        corrs = []
        for cs in motifs:
            for c1 in cs:
                for c2 in cs:
                    if c1 == c2: break
                    if c1 is None or c2 is None: break

                    try:
                        cc, _ = scis.pearsonr(
                            assignments[c1]['intensities'][0],
                            assignments[c2]['intensities'][0])
                        corrs.append(cc)
                    except KeyError:
                        pass

        sns.distplot(
            corrs, ax=ax, hist_kws=dict(alpha=.2), kde=False,
            label='correlations after {} assignment'.format(fname))
        return corrs

    def compute_all_possible_correlations(motifs):
        corrs = []
        for cs in motifs:
            for c1 in cs:
                for c2 in cs:
                    if c1 == c2: break
                    if c1 is None or c2 is None: break

                    for int1 in data[c1]['intensities']:
                        for int2 in data[c2]['intensities']:
                            cc, _ = scis.pearsonr(int1, int2)
                            corrs.append(cc)
        return corrs

    def plot_original_motif_correlations(motifs, ax):
        corrs = compute_all_possible_correlations(motifs)
        sns.distplot(corrs, ax=ax, label='original correlations')

    def plot_idx_list(idx_list, prob_fac, ax):
        ax.plot(idx_list, alpha=.8, label='original index')

        iax = inset_axes(ax, width='30%', height=1., loc=3)
        iax.plot(np.exp(prob_fac*np.arange(50)))
        iax.tick_params(axis='both', which='major', labelsize=5)

    def get_prediction_null_model(motifs, n=3, m=5, num=100):
        """ Draw random intensity vectors and select largest absolute correlation
        """
        # get all intensity vectors
        int_vecs = []
        for cs in motifs:
            for c in cs:
                if c is None: continue
                int_vecs.extend(data[c]['intensities'])

        # compute correlations
        corrs = []
        for _ in trange(num):
            idx_1 = np.random.randint(len(int_vecs), size=n)
            idx_2 = np.random.randint(len(int_vecs), size=m)
            while len(set(idx_1).intersection(idx_2)) > 0:
                idx_2 = np.random.randint(len(int_vecs), size=m)

            int_list_1 = [int_vecs[i] for i in idx_1]
            int_list_2 = [int_vecs[i] for i in idx_2]

            tmp = []
            for int1 in int_list_1:
                for int2 in int_list_2:
                    cc, _ = scis.pearsonr(int1, int2)
                    tmp.append(cc)

            c = max(tmp, key=abs)
            corrs.append(c)

        return corrs

    # plots
    prob_fac = .1
    f, axes = plt.subplots(1, 2, figsize=(21,7))

    all_assignments, all_extra = [], []
    for i in trange(reps):
        assignments, extra = assign(motifs, prob_fac)

        all_assignments.append(assignments)
        all_extra.append(extra)

    ass_corrs = plot_correlations(all_assignments[0], axes[0])
    plot_idx_list(all_extra[0]['idx_list'], prob_fac, axes[1])
    axes[1].plot(all_extra[0]['cur_idx_list'], alpha=.8, label='actual index')
    axes[1].legend(loc='best')

    all_pos_corrs = compute_all_possible_correlations(motifs)
    sns.distplot(
        all_pos_corrs, ax=axes[0], kde=False,
        label='original correlations')
    if null_model:
        sns.distplot(
            get_prediction_null_model(motifs, num=len(ass_corrs)), ax=axes[0],
            label='null model')

    axes[0].legend(loc='best')
    axes[0].set_xlim((-1,1))
    axes[0].set_xlabel('correlation')
    axes[0].set_ylabel('frequency')

    axes[1].set_xlabel('iteration step')
    axes[1].set_ylabel('drawn motif index')

    plt.suptitle(fname)

    #plt.tight_layout()
    plotter.save_figure('images/assignments_{}.pdf'.format(fname), bbox_inches='tight')

    # return assignment results
    return all_assignments, all_extra

def process(compound_data, filter_mz=False):
    """ Simple reaction-combinatorics advancer
    """
    reaction_data = read_reactions_file('data/Reaction_List.csv')
    tmp = iterate_once(compound_data, reaction_data)
    ints = match_masses(tmp)
    out = {k: tmp[k] for k in ints.keys()}
    for k in out: out[k]['intensities'] = ints[k]

    # filter out equal MZ values
    if filter_mz:
        to_del = []
        used_mz = set()
        for c, dat in out.items():
            if dat['mass'] in used_mz:
                to_del.append(c)
            else:
                used_mz.add(dat['mass'])
        for c in to_del:
            del out[c]

    return out

def detect_motifs(graph, motif):
    """ Detect 3-grams in graph isomorphic to motif
    """
    nodes = graph.nodes()
    trips = itertools.product(nodes, nodes, nodes)

    res = []
    for t in tqdm(trips, total=len(nodes)**3):
        if len(set(t)) != 3:
            continue

        sub = graph.subgraph(t)
        if nx.is_isomorphic(sub, motif):
            res.append(sub.edges())

    return res

def detect_ffl(graph):
    """ Detect feedforward loops in graph
        c1 ---> c2
         |       |
         |       v
         -----> c3
    """
    edges = set(graph.edges())
    for c1 in tqdm(graph.nodes()):
        for c2, c3 in itertools.product(
            graph.successors(c1), graph.successors(c1)
        ):
            if len(set([c1,c2,c3])) != 3:
                continue
            if None in (c1, c2, c3):
                continue

            if (c2,c1) in edges or (c3,c1) in edges:
                continue
            if graph.has_edge(c3, c2):
                continue

            if graph.has_edge(c2, c3):
                yield (c1, c2, c3)

def find_more_motifs(motifs, all_compounds, reaction_data, fname='results/post_motif_reactions.pkl'):
    """ Grow fragmented motif network by applying reaction rules to existing ones

        Struture of a motif m:
            (c1, c2, c3, ints, data)
    """
    all_comp_data = dict(pair for d in motifs for pair in d[4].items())
    all_cdata = {}
    for c, group, _ in all_compounds:
        all_cdata[c] = group
    all_cdata.update(all_comp_data)

    all_comp_ints = dict(pair for d in motifs for pair in d[3].items())
    all_cints = {}
    for c, _, ints in all_compounds:
        all_cints.update(ints)
    all_cints.update(all_comp_ints)

    # iterate reactions once
    if not os.path.isfile(fname):
        comps = process(all_cdata)
        with open(fname, 'wb') as fd:
            pickle.dump(comps, fd)
    else:
        print('Using cached data ({})'.format(fname))
        with open(fname, 'rb') as fd:
            comps = pickle.load(fd)

    # transform result into more usable form
    tmp = collections.defaultdict(list)
    for p, data in tqdm(comps.items()):
        c1, c2 = data['origin'][0], data['origin'][1]
        tmp[c1].append(p)
        tmp[c2].append(p)

    # grow motif network
    new_links = []
    old_links = []
    for m in tqdm(motifs):
        for c in m[:3]:
            if c in tmp:
                for t in tmp[c]:
                    new_links.append((c, t))
        old_links.extend([
            (m[0], m[1]), (m[0], m[2]), (m[1], m[2])
        ])

    # find motifs in graph
    graph = nx.DiGraph()
    graph.add_edges_from(old_links)
    graph.add_edges_from(new_links)

    more_motifs = []
    for c1,c2,c3 in detect_ffl(graph):
        res = {'data': {}, 'ints': {}}
        for c in (c1,c2,c3):
            assert c in all_cdata or c in comps
            if c in all_cdata:
                res['data'][c] = all_cdata[c]
                res['ints'][c] = all_cints[c]
            else:
                res['data'][c] = comps[c]
                #del res['data'][c]['intensities']
                res['ints'][c] = comps[c]['intensities']

            assert len(res['ints'][c]) > 0

        more_motifs.append((c1, c2, c3, res['ints'], res['data']))

        # final checks
        try:
            assert isinstance(more_motifs[-1][0], str)
            assert isinstance(more_motifs[-1][1], str)
            assert isinstance(more_motifs[-1][2], str)
            assert isinstance(more_motifs[-1][3], dict)
            assert isinstance(more_motifs[-1][4], dict)

            for e in more_motifs[-1][3].values():
                assert isinstance(e, list)
        except AssertionError:
            import ipdb; ipdb.set_trace()

    return more_motifs

def plot_mz_distribution(motifs, data, fname='data/peaklist_filtered_assigned.csv'):
    """ Plot MZ values of data and highlight real-life entries
    """
    # create set of motifs
    comp_in_motif = set(itertools.chain(*motifs))

    # aggregate data
    mzs = []
    single_matches = []
    for name, info in data.items():
        if not name in comp_in_motif:
            continue

        mz = info['mass']
        mzs.append(mz)

        assert len(info['intensities']) > 0
        if len(info['intensities']) == 1:
            single_matches.append(mz)

    peak_data = read_peak_data(fname)

    # plot
    fig = plt.figure()
    plt.hist(mzs, 100, alpha=0.7, linewidth=0,)

    for mz, _ in peak_data.items():
        plt.axvline(mz, color='red', alpha=0.03)
    for mz in single_matches:
        plt.axvline(mz, color='green', alpha=0.02)

    plt.xlabel('MZ value')
    plt.ylabel('count')
    plt.title('MZ histogram of motif compounds')

    plt.tight_layout()
    plotter.save_figure('images/rl_mz_hist.pdf', bbox_inches='tight')

def plot_intensity_number_distribution(comps):
    data = [len(v['intensities']) for v in comps.values()]

    plt.figure()
    plt.hist(data, bins=range(max(data)))
    plt.title('Intensity number distribution')
    plt.xlabel('number of intensity assignments per compound')
    plt.savefig('images/intpcomp_distr.pdf')

def get_origin_set(comp, data):
    """ Find basic origins which comp is made out of
    """
    if comp is None:
        return set()

    if data[comp]['origin'] == (None, None):
        return {comp}

    return get_origin_set(data[comp]['origin'][0], data) | get_origin_set(data[comp]['origin'][1], data)

def compare_assignment_result(ass_data, data):
    """ Check robustness of comparison by counting how many assignments vary over multiple runs
    """
    fig, axes = plt.subplots(1, len(ass_data), figsize=(20,5))

    for (all_ass, lbl), ax in zip(ass_data, axes):
        # aggregate assignments over various runs
        tmp = {'run': [], 'intensity_vec': [], 'compound': []}
        for i, assignments in enumerate(all_ass):
            # make intensity vectors hashable
            ass = {k: tuple(v) for k,v in assignments.items()}

            for comp, ints in ass.items():
                tmp['run'].append(i)
                tmp['intensity_vec'].append(ints)
                tmp['compound'].append(comp)
        df = pd.DataFrame(tmp)

        # pre-filter (only consider compounds which got always mapped)
        tmp = []
        for name, group in list(df.groupby('run')):
            cs = group['compound'].tolist()
            tmp += cs
        counts = pd.Series(tmp).value_counts()
        needed_comps = counts[counts==(df['run'].max()+1)].index
        df = df[df['compound'].isin(needed_comps)]

        # check robustness
        int_res = []
        for name, group in df.groupby('intensity_vec'):
            comp_vec = group['compound'].unique()

            # filter entries which have only one assignment anyways
            assert len(comp_vec) >= 1
            if len(comp_vec) == 1:
                if len(data[list(comp_vec)[0]]['intensities']) == 1:
                    assert tuple(data[list(comp_vec)[0]]['intensities'][0]) == name
                    continue
            int_res.append(len(comp_vec)==1)
        int_val = sum(int_res) / len(int_res) if len(int_res) > 0 else 0

        comp_res = []
        for name, group in df.groupby('compound'):
            int_vec = group['intensity_vec'].unique()

            # filter entries which have only one assignment anyways
            assert len(int_vec) >= 1
            if len(data[name]['intensities']) == 1:
                assert tuple(data[name]['intensities'][0]) == int_vec[0]
                continue
            comp_res.append(len(int_vec)==1)
        comp_val = sum(comp_res) / len(comp_res) if len(comp_res) > 0 else 0

        # plot
        ax.bar(
            -.5, int_val, width=.5,
            label='len(comp_vec)==1', color='b')
        ax.bar(
            0, comp_val, width=.5,
            label='len(int_vec)==1', color='g')
        ax.legend(loc='best')

        ax.set_xlim((-1, 1))
        ax.set_ylim((0, 1))

        title = f'''{lbl}
{round(int_val, 2)} ({sum(int_res)}/{len(int_res)}/{len(df.intensity_vec.unique())})
{round(comp_val, 2)} ({sum(comp_res)}/{len(comp_res)}/{len(df["compound"].unique())})'''
        ax.set_title(title)
        print(title)

    plt.tight_layout()
    plt.savefig('images/assignment_comparison.pdf')

def compare_to_realdata(ass_data, input_data):
    """ Check assignments via comparison to real-life data
    """
    def find_true_mz(ints, pdata):
        for p,i in pdata.items():
            if i == ints:
                return p
        return None

    def convert_assignments(ass, pdata):
        tmp = {'name': [], 'formula': [], 'mz': []}
        for c,data in ass.items():
            assert len(data['intensities']) == 1
            tmz = find_true_mz(data['intensities'][0], pdata)

            tmp['name'].append(c)
            tmp['formula'].append(gen_atom_string(data['atoms']))
            tmp['mz'].append(tmz)

        return pd.DataFrame(tmp)

    def remove_trivial_assignments(ass, pdata):
        """ Remove cases which initially only have one assignment possibility
        """
        tmp = dict(ass)
        current_compounds = list(ass.keys())
        for comp in current_compounds:
            cur = input_data[comp]

            assert len(cur['intensities']) > 0
            if len(cur['intensities']) == 1:
                tmp.pop(comp)

        #print(f'Trivial assignment removal: {len(ass)}->{len(tmp)}')
        return tmp

    comp_df = formula_investigator.get_rl_comparison_frame()
    pdata = read_peak_data('data/peaklist_filtered_assigned.csv')

    plt.figure(figsize=(6, 4*len(ass_data)))
    ax = None
    for i, (all_ass, all_info, lbl) in enumerate(tqdm(ass_data)):
        cur_dists = []
        for assignments, info in tqdm(zip(all_ass, all_info), total=len(all_ass)):
            ass_tmp = remove_trivial_assignments(assignments, pdata)
            new_ass = convert_assignments(ass_tmp, pdata)
            match = new_ass.merge(comp_df, left_on='name', right_on='cname')
            cur_dists.extend(match['dist'].tolist())
        cur_dists = np.asarray(cur_dists)

        # compute quality
        qual_vals = {x: (cur_dists<=x).sum()/cur_dists.size for x in [5,10,50]}

        # plot distance distribution
        ax = plt.subplot(len(ass_data), 1, i+1, sharex=ax, sharey=ax)
        sns.distplot(cur_dists, bins=30, kde=False, norm_hist=True)

        plt.title(f'{lbl} (quality: {qual_vals[5]:.2},{qual_vals[10]:.2},{qual_vals[50]:.2})')
        plt.xlabel('formula distance')
        plt.ylabel('count')

    plt.tight_layout()
    plt.savefig('images/formdist_distr.pdf')

def investigate_prediction_chaos(ass_data, data):
    """ Check how inter-connectedness influences variability of prediction result

        * Count how often each node pair is assigned to particular intensity-vector pair
    """
    print('Chaos plots')
    fig, axes = plt.subplots(len(ass_data), 3)

    for (all_ass, all_info, lbl), ax_row in zip(ass_data, axes):
        # aggregate assignments over various runs
        df = pd.DataFrame()
        assert len(all_ass) == len(all_info)
        tmp = {lbl: [] for lbl in ['run', 'intensity_vec', 'compound', 'compound_assignment_idx']}

        for i, (assignments, info) in enumerate(zip(all_ass, all_info)):
            for comp, ints in assignments.items():
                assert comp in info['assignment_order']
                tmp['run'].append(i)
                tmp['intensity_vec'].append(tuple(ints))
                tmp['compound'].append(comp)
                tmp['compound_assignment_idx'].append(info['assignment_order'].index(comp))
        df = pd.DataFrame(tmp)

        # precompute some statistic
        comp_frac = {}
        for name, group in df.groupby('compound'):
            int_vec = group['intensity_vec'].unique()

            # filter entries which have only one assignment anyways
            assert len(int_vec) >= 1
            if len(data[name]['intensities']) == 1:
                assert tuple(data[name]['intensities'][0]) == int_vec[0]
                comp_frac[name] = -1
                continue

            assert not name in comp_frac
            comp_frac[name] = len(int_vec)

        # make scatter plots
        for compound, group in tqdm(df.groupby('compound')):
            int_val = comp_frac[compound]
            mean_idx = group['compound_assignment_idx'].mean()
            std_idx = group['compound_assignment_idx'].std()
            avg_intvec_len = len(group['intensity_vec'].unique())

            ax_row[0].scatter(
                int_val, mean_idx,
                alpha=.3, rasterized=True)
            ax_row[1].scatter(
                mean_idx, std_idx,
                alpha=.3, rasterized=True)
            ax_row[2].scatter(
                mean_idx, avg_intvec_len,
                alpha=.3, rasterized=True)

        ax_row[0].set_title(lbl)
        ax_row[0].set_xlabel('int_val')
        ax_row[0].set_ylabel('mean_idx')

        ax_row[1].set_title(lbl)
        ax_row[1].set_xlabel('mean_idx')
        ax_row[1].set_ylabel('std_idx')

        ax_row[2].set_title(lbl)
        ax_row[2].set_xlabel('mean_idx')
        ax_row[2].set_ylabel('avg_intvec_len')

    plt.tight_layout()
    zoom = 1.5
    w, h = fig.get_size_inches()
    fig.set_size_inches(w * zoom, h * zoom)
    plt.savefig('images/assignment_chaos.pdf')

def null_model_assignments(data, num, reps=100):
    """ Choose random assignment per compound
    """
    def nm_assign(num, compounds):
        np.random.shuffle(compounds)
        assignments = {}
        for node_sel in compounds:
            # select random assignment
            cur_ints = data[node_sel]['intensities'][:]
            for i in assignments.values():
                if i in cur_ints:
                    cur_ints.remove(i)

            if len(cur_ints) == 0:
                continue
            int_sel = random.choice(cur_ints)

            # finalize
            assignments[node_sel] = {
                'intensities': [int_sel],
                'atoms': data[node_sel]['atoms']
            }
        return assignments, {'assignment_order': compounds[:]}

    # choose random compounds
    compounds = []
    all_compounds = list(data.keys())
    for _ in range(num):
        node_sel = random.choice(all_compounds)
        while node_sel in compounds:
            node_sel = random.choice(all_compounds)
        compounds.append(node_sel)
    assert len(set(compounds)) == len(compounds) # unique compounds

    return zip(*[nm_assign(num, compounds) for _ in trange(reps)])

def find_small_motifs(
    compounds_level0,
    fname='cache/rf_raw_reaction_data.pkl'
):
    """ Look for feedfoward-loops in (iterated) compound data
    """
    # let compounds react
    if not os.path.isfile(fname):
        comps = {}
        comps.update(compounds_level0)

        for i in range(2):
            tmp = process(comps)
            tqdm.write('Found {} new compounds [#{}]'.format(
                len(tmp), i))
            comps.update(tmp)

        with open(fname, 'wb') as fd:
            pickle.dump(comps, fd)
    else:
        print('Using cached data ({})'.format(fname))
        with open(fname, 'rb') as fd:
            comps = pickle.load(fd)

    print('Proceeding with {} compounds'.format(len(comps)))

    # grow network
    graph = nx.DiGraph()
    for p, data in tqdm(comps.items()):
        c1, c2 = data['origin'][0], data['origin'][1]
        if None in (c1,c2,p): continue
        graph.add_edge(c1, p)
        graph.add_edge(c2, p)

    # find motifs
    # Note: maybe weight them according to occurences of same compound configuration with different reactions
    motifs = []
    for cs in detect_ffl(graph):
        if None in cs: continue
        if cs in motifs: continue
        motifs.append(cs)

    print('Found {} motifs'.format(len(motifs)))

    # conduct predictions
    print('Predicting')
    other_size = len(motifs)*3

    # predictions with motifs
    motif_fname = 'cache/prediction_motif.dat'
    if not os.path.exists(motif_fname):
        motif_ass, motif_info = find_optimal_assignments(motifs, comps)

        with open(motif_fname, 'wb') as fd:
            pickle.dump({
                'ass': motif_ass,
                'info': motif_info
            }, fd)
    else:
        print('Using cached data ({})'.format(motif_fname))
        with open(motif_fname, 'rb') as fd:
            tmp = pickle.load(fd)
            motif_ass = tmp['ass']
            motif_info = tmp['info']

    # predict using links from motif network
    motiflinks_fname = 'cache/prediction_motiflinks.dat'
    if not os.path.exists(motiflinks_fname):
        motiflinks = [edge
            for c1,c2,c3 in motifs
                for edge in [(c1,c2,None),(c2,c3,None),(c1,c3,None)]]
        motiflink_ass, motiflink_info = find_optimal_assignments(
            motiflinks, comps, fname='motiflinks')

        with open(motiflinks_fname, 'wb') as fd:
            pickle.dump({
                'ass': motiflink_ass,
                'info': motiflink_info
            }, fd)
    else:
        print('Using cached data ({})'.format(motiflinks_fname))
        with open(motiflinks_fname, 'rb') as fd:
            tmp = pickle.load(fd)
            motiflink_ass = tmp['ass']
            motiflink_info = tmp['info']

    # predict using only links
    links_fname = 'cache/prediction_links.dat'
    if not os.path.exists(links_fname):
        edge_idx = np.random.choice(
            np.arange(len(graph.edges())), size=other_size)
        links = [(*graph.edges()[edx],None) for edx in edge_idx]
        link_ass, link_info = find_optimal_assignments(links, comps, fname='links')

        with open(links_fname, 'wb') as fd:
            pickle.dump({
                'ass': link_ass,
                'info': link_info
            }, fd)
    else:
        print('Using cached data ({})'.format(links_fname))
        with open(links_fname, 'rb') as fd:
            tmp = pickle.load(fd)
            link_ass = tmp['ass']
            link_info = tmp['info']

    # predict using random nodes
    random_fname = 'cache/prediction_random.dat'
    if not os.path.exists(random_fname):
        node_sel = [n
            for n in graph.nodes()
                if len(comps[n]['intensities']) >= 5]
        rand_nodes = list(set(
            [(*np.random.choice(node_sel, size=2),None)
                for _ in range(other_size)]
        ))
        random_ass, random_info = find_optimal_assignments(rand_nodes, comps, fname='random')

        with open(random_fname, 'wb') as fd:
            pickle.dump({
                'ass': random_ass,
                'info': random_info
            }, fd)
    else:
        print('Using cached data ({})'.format(random_fname))
        with open(random_fname, 'rb') as fd:
            tmp = pickle.load(fd)
            random_ass = tmp['ass']
            random_info = tmp['info']

    # use another null-model
    null_ass, null_info = null_model_assignments(comps, other_size*2)

    # compare assignment results
    compare_to_realdata([
        (motif_ass, motif_info, 'motifs'),
        (motiflink_ass, motiflink_info, 'motiflinks'),
        (link_ass, link_info, 'links'),
        (random_ass, random_info, 'random'),
        (null_ass, null_info, 'nullmodel')
    ], comps)
    investigate_prediction_chaos([
        (motif_ass, motif_info, 'motifs'),
        (motiflink_ass, motiflink_info, 'motiflinks'),
        (link_ass, link_info, 'links'),
        (random_ass, random_info, 'random'),
        (null_ass, null_info, 'nullmodel')
    ], comps)
    compare_assignment_result([
        (motif_ass, 'motifs'),
        (motiflink_ass, 'motiflinks'),
        (link_ass, 'links'),
        (random_ass, 'random'),
        (null_ass, 'nullmodel')
    ], comps)

    ## plot stuff
    print('Plotting')
    plot_network(motifs, comps)
    plot_intensity_number_distribution(comps)
    plot_mz_distribution(motifs, comps)

    # random compounds for comparison
    unrelated_nodes = []

    # only use MZ values from motifs
    mzs = []
    for trip in motifs:
        for c in trip:
            mzs.append(comps[c]['mass'])
    max_mz = max(mzs)
    min_mz = min(mzs)

    nodes = []
    for n in graph.nodes():
        if n is None: continue
        if comps[n]['mass'] > min_mz and comps[n]['mass'] < max_mz:
            nodes.append(n)

    es = lambda f, s: len(f.intersection(s)) == 0 # check for empty set intersection
    while len(unrelated_nodes) < len(motifs):
        # *blargh*
        first = np.random.choice(nodes)
        second = np.random.choice(nodes)
        while not es(get_origin_set(second, comps), get_origin_set(first, comps)):
            second = np.random.choice(nodes)
        third = np.random.choice(nodes)
        while not es(get_origin_set(third, comps), get_origin_set(first, comps)) or not es(get_origin_set(third, comps), get_origin_set(second, comps)):
            third = np.random.choice(nodes)

        th = (first, second, third)
        if not th in unrelated_nodes:
            unrelated_nodes.append(th)

    plot_correlation_histogram((
            (motifs, 'FFL'),
            (unrelated_nodes, 'disconnected')
        ), comps)

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

    #peak_data = read_peak_data('data/peaklist_filtered_assigned.csv')

    # combine data
    intensities_all = {}
    intensities_all.update(intensities_level0)
    intensities_all.update(intensities_level1)

    compounds_all = {}
    compounds_all.update(compounds_level0)
    compounds_all.update(compounds_level1)

    # aggregate data
    df_data = []
    for name, data in compounds_all.items():
        df_data.append({
            'mz': data['mass'],
            'formula': gen_atom_string(data['atoms'])
        })

    df = pd.DataFrame.from_dict(df_data)
    print(df.head())
    df.to_csv('data/mz_formula.csv')

def main(compound_fname, reaction_fname):
    """ Read in data and start experiment
    """
    compound_data = read_compounds_file(compound_fname)
    reaction_data = read_reactions_file(reaction_fname)

    # only keep masses which have associated intensities
    intensities_level0 = match_masses(compound_data)
    compounds_level0 = {k: compound_data[k] for k in intensities_level0.keys()}

    for k in compounds_level0:
        compounds_level0[k]['intensities'] = intensities_level0[k]

    # find new compounds
    print('Starting with {} compounds'.format(len(compounds_level0)))

    # investigate results
    #list_structure_formulas(
    #    compounds_level0, intensities_level0,
    #    reaction_data)

    #investigate_reactions(
    #    compounds_level0, intensities_level0,
    #    reaction_data)

    find_small_motifs(compounds_level0)


if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    main('data/Compound_List.csv', 'data/Reaction_List.csv')
