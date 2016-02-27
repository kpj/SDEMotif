from unittest import TestCase

import numpy as np
import numpy.testing as npt

from utils import *


class TestSigEntryExtraction(TestCase):
    def test_valid_matrix(self):
        mat = np.array([
            [1, 2, 3],
            [2, 1, 4],
            [3, 4, 1]
        ])
        res = extract_sig_entries(mat)
        npt.assert_array_equal(res, np.array([2, 3, 4]))

    def test_too_small_matrix(self):
        small_mat = np.array([
            [1, 2],
            [2, 1],
        ])
        with self.assertRaises(AssertionError):
            extract_sig_entries(small_mat)

    def test_non_symmetric_matrix(self):
        non_sym_mat = np.array([
            [1, 2, 5],
            [2, 1, 4],
            [3, 4, 1]
        ])
        with self.assertRaises(AssertionError):
            extract_sig_entries(non_sym_mat)
