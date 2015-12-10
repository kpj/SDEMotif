"""
Visualization related functions
"""

import scipy.stats as scis

import matplotlib.pylab as plt


def plot_system_evolution(sol):
    """ Plot solution of integration
    """
    for i, series in enumerate(sol):
        plt.plot(series, label=r'$S_%d$' % (i+1))

    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend(loc='best')

    plt.show()

def plot_ss_scatter(steadies):
    """ Plot scatter plots of steady states
    """
    def do_scatter(i, j, ax):
        """ Draw single scatter plot
        """
        xs, ys = steadies[:,i], steadies[:,j]

        ax.scatter(xs, ys)

        ax.set_xlabel(r'$S_%d$' % (i+1))
        ax.set_ylabel(r'$S_%d$' % (j+1))

        cc, pval = scis.pearsonr(xs, ys)
        ax.set_title(r'Corr: $%.2f$' % cc)

    dim = steadies.shape[1]
    fig, axarr = plt.subplots(1, dim)

    axc = 0
    for i in range(dim):
        for j in range(dim):
            if i == j: break
            do_scatter(
                i, j,
                axarr[axc])
            axc += 1

    plt.suptitle('Correlation overview')

    plt.show()
