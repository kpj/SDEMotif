"""
Investigate chemical formulas
"""

import pickle
import itertools

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

import reaction_finder


def read_combinatorial_compounds(fname='cache/rf_raw_reaction_data.pkl'):
    with open(fname, 'rb') as fd:
        comps = pickle.load(fd)

    tmp = {'Name': [], 'Formula': [], 'MZ': []}

    for name, data in comps.items():
        tmp['Name'].append(name)
        tmp['Formula'].append(reaction_finder.gen_atom_string(data['atoms']))
        tmp['MZ'].append(data['mass'])

    return pd.DataFrame(tmp)

def read_actual_compounds(fname='data/Res_Polphen_List_Neg.csv'):
    #sed 's/\([[:digit:]]\),\([[:digit:]]\)/\1.\2/g' data/Res_Polphen_List_Neg.csv
    df = pd.read_csv(fname)
    df = df[['Name', 'Formula', 'M_selected']]
    df.rename(columns={'M_selected': 'MZ'}, inplace=True)
    return df

def merge_sources(df_comb, df_roy, thres=.01):
    tmp = {'aname': [], 'cname': [], 'aform': [], 'cform': [], 'amass': [], 'cmass': []}
    for row in tqdm(df_comb.itertuples(), total=df_comb.shape[0]):
        match = df_roy[abs(row.MZ-df_roy['MZ']) < thres]

        if match.empty:
            continue

        if match.shape[0] == 1:
            tmp['aname'].append(match.iloc[0].Name)
            tmp['cname'].append(row.Name)
            tmp['aform'].append(match.iloc[0].Formula)
            tmp['cform'].append(row.Formula)
            tmp['amass'].append(match.iloc[0].MZ)
            tmp['cmass'].append(row.MZ)
        else:
            for r in match.itertuples():
                tmp['aname'].append(r.Name)
                tmp['cname'].append(row.Name)
                tmp['aform'].append(r.Formula)
                tmp['cform'].append(row.Formula)
                tmp['amass'].append(r.MZ)
                tmp['cmass'].append(row.MZ)

    return pd.DataFrame(tmp)

def form2dict(form):
    """ Convert single formula to a dict describing its composition
    """
    result = {}

    cur_atom = None
    cur_num = ''
    for c in form:
        if c.isdigit():
            assert cur_atom is not None
            cur_num += c
        else:
            if cur_atom is None:
                cur_atom = c
            else: # atoms have single-char names in our case
                result[cur_atom] = int(cur_num) if len(cur_num) > 0 else 1
                cur_atom = c
                cur_num = ''
    result[cur_atom] = int(cur_num) if len(cur_num) > 0 else 1

    return result

def compute_formula_distance(form1, form2):
    """ Approximate way of comparing two formulas
    """
    res1 = form2dict(form1)
    res2 = form2dict(form2)
    all_atoms = set.union(set(res1.keys()), set(res2.keys()))

    diff = sum([abs(res1.get(a,0)-res2.get(a,0)) for a in all_atoms])
    total = sum(res1.values()) + sum(res2.values())

    return diff / total

def get_rl_comparison_frame():
    """ Return merged DataFrame of actual and combinatorial data
    """
    com_data = read_combinatorial_compounds()
    act_data = read_actual_compounds()

    df = merge_sources(com_data, act_data)
    df['dist'] = df.apply(
        lambda row: compute_formula_distance(row.aform, row.cform),
        axis=1)
    return df

def mz_range_comparison():
    """ Visualize MZ-value ranges
    """
    com_data = read_combinatorial_compounds()
    act_data = read_actual_compounds()

    plt.figure()

    sns.distplot(com_data['MZ'], label='combinatorial data')
    sns.distplot(act_data['MZ'], label='roy data')

    plt.xlabel('mz value')

    plt.legend(loc='best')
    plt.savefig('images/mz_ranges.pdf')

def plot_formdist_nullmodel():
    """ Plot formula distances in given data
    """
    # read data
    df = read_actual_compounds()
    forms = itertools.product(df['Formula'].tolist(), repeat=2)

    # compute result
    dists_all = []
    for f1, f2 in forms:
        dist = compute_formula_distance(f1, f2)
        dists_all.append(dist)

    dists_matched = get_rl_comparison_frame()['dist'].tolist()

    # plot result
    plt.figure()

    sns.distplot(dists_all, label='all real data', kde=False)
    sns.distplot(dists_matched, label='matched data', kde=False)

    plt.legend(loc='best')
    plt.savefig('images/formdist_nullmodel.pdf')

def main():
    mz_range_comparison()
    df = get_rl_comparison_frame()

    print(df.sort_values('dist')[['aname', 'cname', 'dist']].head(20))
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
