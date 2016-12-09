"""
Clean pipeline of robustness detection
"""

from typing import Tuple

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

def initial_tests(n=1000: int) -> None:
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

def main() -> None:
    sns.set_style('white')
    plt.style.use('seaborn-poster')

    initial_tests()

if __name__ == '__main__':
    main()
