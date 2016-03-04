"""
General data filters.
An entry is filtered if a filter returns True
"""

import numpy as np


def filter_steady_state(ss, filter_mask, min_thres=1e-10, max_thres=20):
    """ Discard steady states which are trivial (~= `min_thres`) or diverge (> `max_thres`)
    """
    mask = np.ones(ss.shape, dtype=bool)
    if not filter_mask is None:
        for ind in filter_mask:
            try:
                mask[ind] = 0
            except IndexError:
                pass

    fss = ss[mask]
    return any(fss < min_thres) or any(fss > max_thres)
