from unittest import TestCase

import numpy as np
import numpy.testing as npt

from utils import *


class TestFunctions(TestCase):
    def test_extract_sig_entries(self):
        mat = np.array([
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1]
        ])
        res = extract_sig_entries(mat)
        npt.assert_array_equal(res, np.array([2, 3, 4]))

        small_mat = np.array([
            [1, 2],
            [2, 1],
        ])
        with self.assertRaises(AssertionError):
            extract_sig_entries(small_mat)

        non_sym_mat = np.array([
            [1, 2, 5],
            [2, 1, 4],
            [3, 4, 1]
        ])
        with self.assertRaises(AssertionError):
            extract_sig_entries(non_sym_mat)
