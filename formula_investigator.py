"""
Investigate chemical formulas
"""

import pickle

import pandas as pd
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

def merge_sources(df1, df2, thres=.01):
    tmp = {'aname': [], 'cname': [], 'aform': [], 'cform': [], 'amass': [], 'cmass': []}
    for row in tqdm(df1.itertuples(), total=df1.shape[0]):
        match = df2[abs(row.MZ-df2['MZ']) < thres]

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
    return diff

def get_rl_comparison_frame():
    com_data = read_combinatorial_compounds()
    act_data = read_actual_compounds()

    df = merge_sources(com_data, act_data)
    df['dist'] = df.apply(
        lambda row: compute_formula_distance(row.aform, row.cform),
        axis=1)
    return df

def main():
    df = get_rl_comparison_frame()

    print(df.sort_values('dist')[['aname', 'cname', 'dist']].head(20))
    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
