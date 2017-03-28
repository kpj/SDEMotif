"""
General data filters.
An entry is filtered if a filter returns True
"""

import numpy as np


def filter_steady_state(ss, filter_mask=None, min_thres=1e-10, min_inc=1e-1):
    """ Discard steady states which are trivial (~= `min_thres`) or diverge (differences increase by at least `min_inc`)
    """
    fss = np.copy(ss)
    if not filter_mask is None:
        for ind in filter_mask:
            try:
                fss = np.delete(fss, ind, axis=1)
            except IndexError:
                pass

    # check divergence
    old_diff = abs(fss[1] - fss[0])
    new_diff = abs(fss[-1] - fss[-2])
    inc_diff = new_diff >= old_diff
    div_res = inc_diff.any() and (new_diff > min_inc).any()

    # check convergence to 0
    conv_res = (fss < min_thres).any()

    return conv_res or div_res
