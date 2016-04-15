from unittest import TestCase

import numpy.testing as npt
import pandas as pd

from network_space import *


class TestFunction(TestCase):
    def test_check_vanishing(self):
        res = check_vanishing(pd.DataFrame({
            'raw_res': None,
            'enh_res': None,
            'raw_vals': [0.8, 0.9, 0],
            'enh_vals': [0, 0.3, 0]
        }))

        self.assertEqual(len(res), 4)
        self.assertEqual(res.already_zero, 1)
        npt.assert_almost_equal(res.abs_corr_diff, [0.6])
