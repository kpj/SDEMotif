"""
Investigate chemical formulas
"""

import pickle

import pandas as pd
from tqdm import tqdm

from reaction_finder import gen_atom_string


def read_combinatorial_compounds(fname='cache/rf_raw_reaction_data.pkl'):
    with open(fname, 'rb') as fd:
        comps = pickle.load(fd)

    tmp = {'Name': [], 'Formula': [], 'MZ': []}

    for name, data in comps.items():
        tmp['Name'].append(name)
        tmp['Formula'].append(gen_atom_string(data['atoms']))
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

def main():
    com_data = read_combinatorial_compounds()
    act_data = read_actual_compounds()

    df = merge_sources(com_data, act_data)

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
