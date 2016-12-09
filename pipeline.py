"""
Clean pipeline of robustness detection
"""

import sys
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm


def threshold_div(distr: np.ndarray, thres: float) -> Tuple[np.ndarray, np.ndarray]:
    """ Divide distribution into below `-thres` and above `thres`
    """
    below = distr[distr<-thres]
    above = distr[distr>thres]
    return below, above

def compare_distributions(dis1: np.ndarray, dis2: np.ndarray, thres: float) -> float:
    """ Compare two distributions according to robustness measure
    """
    c_1m, c_1p = threshold_div(dis1, thres)
    c_2m, c_2p = threshold_div(dis2, thres)

    n_1m, n_1p = c_1m.size/dis1.size, c_1p.size/dis1.size
    n_2m, n_2p = c_2m.size/dis2.size, c_2p.size/dis2.size

    return n_1p * n_2m * (n_2m - n_1m) + n_1m * n_2p * (n_2p - n_1p)

def initial_tests(n: int = 1000) -> None:
    """ Conduct some initial tests with robustness measure
    """
    dis_3_x = -.08
    dis_3 = np.random.normal(dis_3_x, 0.1, size=n)

    # generate data
    thresholds = np.logspace(-4, -1, 15)
    data = {t: [] for t in thresholds}
    data.update({'x': []})
    for x in tqdm(np.linspace(-1, 1, 100)):
        dis_4 = np.random.normal(x, 0.1, size=n)

        data['x'].append(x)
        for t in thresholds:
            rob = compare_distributions(dis_3, dis_4, t)
            data[t].append(rob)

    df = pd.DataFrame(data)
    df.set_index('x', inplace=True)

    # plot data
    df.plot(legend=False)

    plt.title(r'First distribution: $x={}$'.format(dis_3_x))
    plt.xlabel('x-pos of second distribution')
    plt.ylabel('magic quantity')

    plt.savefig('images/robustness_development.pdf')

def handle_enh_entry(raw, enh, thres: float) -> float:
    """ Compare distribution from given simulation results
    """
    return 1

def threshold_influence(data: List, resolution: int = 5) -> None:
    """ Plot robustness for varying threshold levels
    """
    threshold_list = np.logspace(-5, 0, resolution-1)

    # generate data
    df = pd.DataFrame()
    for thres in tqdm(threshold_list):
        cur = []
        for raw, enh_res in data:
            res = [handle_enh_entry(raw, enh, thres) for enh in enh_res]
            cur.append(res)
        df = df.append({
            'threshold': thres,
            'robustness': np.sum(cur)
        }, ignore_index=True)

    # plot data
    plt.figure()

    df.plot(
        'threshold', 'robustness',
        kind='scatter', loglog=True,
        ax=plt.gca())

    plt.savefig('images/threshold_influence.pdf')

def main(fname) -> None:
    with open(fname, 'rb') as fd:
        inp = pickle.load(fd)

    threshold_influence(np.asarray(inp['data']))

if __name__ == '__main__':
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    if len(sys.argv) == 1:
        print('No arguments given, running tests')
        initial_tests()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        print('Usage: {} [data file]'.format(sys.argv[0]))
        exit(-1)
