"""
Useful functions and more
"""


def get_nonconst_data(i, j, data):
    """ Return data along some axis and make sure it's not constant
    """
    xs, ys = data[:,i], data[:,j]

    # fix series with stdev == 0
    xs[0] += 1e-20
    ys[0] += 1e-20

    return xs, ys
