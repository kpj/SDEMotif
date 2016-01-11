"""
General data filters
"""

import numpy as np


def filter_steady_state(ss, min_thres=1e-10, max_thres=20):
    """ Discard steady states which are trivial (~= `min_thres`) or diverge (> `max_thres`)
    """
    return any(ss < min_thres) or any(ss > max_thres)

def filter_correlation_matrix(corr_mat):
    """ Filter matrices with zero entries
    """
    return np.count_nonzero(corr_mat) != corr_mat.size
