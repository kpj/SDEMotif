from unittest import TestCase

from filters import *


class TestSteadyStateFilter(TestCase):
    def test_convergence_to_zero(self):
        ss = np.array([
            [1, 1e-9, 1, 1],
            [2, 1e-10, 1, 2],
            [1, 1e-11, 1, 2]
        ])

        res = filter_steady_state(ss, None)
        self.assertTrue(res)

    def test_masked_convergence_to_zero(self):
        ss = np.array([
            [1, 1e-9, 1, 1],
            [2, 1e-10, 1, 2],
            [1, 1e-11, 1, 2]
        ])

        res = filter_steady_state(ss, [1])
        self.assertFalse(res)

    def test_divergence(self):
        ss = np.array([
            [1, 1e3],
            [2, 1e4],
            [1, 1e5]
        ])

        res = filter_steady_state(ss, None)
        self.assertTrue(res)

    def test_oob_mask(self):
        ss = np.array([
            [1, 1e-9, 1, 1],
            [2, 1e-10, 1, 2],
            [1, 1e-11, 1, 2]
        ])

        res = filter_steady_state(ss, [4])
        self.assertTrue(res)
