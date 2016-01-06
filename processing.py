"""
Data processing facilities
"""

import collections

import numpy as np
import matplotlib.pylab as plt


def network_density():
    """ Plot network edge density vs correlation quotient
    """
    data = np.load('results/data_cache_addnode.npy')

    points = collections.defaultdict(list)
    for syst, mat, sol in data:
        max_edge_num = syst.jacobian.shape[0] * (syst.jacobian.shape[0]-1)
        dens = np.count_nonzero(syst.jacobian) / max_edge_num
        quot = mat[0, 2] / mat[1, 2]
        points[dens].append(quot)

    densities = []
    quotients = []
    errbars = []
    for dens, quots in points.items():
        densities.append(dens)
        quotients.append(np.mean(quots))
        errbars.append(np.std(quots))

    plt.errorbar(
        densities, quotients, yerr=errbars,
        fmt='o', clip_on=False)

    plt.xlabel('motif edge density')
    plt.ylabel('correlation quotient')
    #plt.xlim((0.5, 1.01))

    plt.tight_layout()
    plt.savefig('images/edens_quot.pdf', bbox_inches='tight')
    plt.show()

def main():
    """ Main interface
    """
    network_density()


if __name__ == '__main__':
    main()
