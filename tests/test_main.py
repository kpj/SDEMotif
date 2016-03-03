from unittest import TestCase

import numpy as np
import numpy.testing as npt

from main import *
from setup import generate_basic_system
from system import SDESystem


class TestSteuerSystem(TestCase):
    """ Check results given in paper
    """
    def test_run(self):
        steuer_syst = generate_basic_system()
        steuer_syst.fluctuation_vector = np.array([1e-5, 0, 0]) # reduce randomness

        syst, mat, sol = analyze_system(
            steuer_syst,
            repetition_num=100)

        self.assertEqual(steuer_syst, syst)

        npt.assert_allclose(sol[0][-1], 2.5, atol=0.1)
        npt.assert_allclose(sol[1][-1], 1.25, atol=0.1)
        npt.assert_allclose(sol[2][-1], 5., atol=0.1)

        npt.assert_allclose(mat, np.array([
            [1, 0.7, 0.47],
            [0.7, 1, 0.87],
            [0.47, 0.87, 1]
        ]), atol=0.3)

class TestSimulationFilters(TestCase):
    def setUp(self):
        J = np.array([[0, 0],[0, 0]])
        D_E = np.array([0, 0])
        init = np.array([1, 1])
        self.syst = SDESystem(J, D_E, D_E, init)

    def test_convergence_to_zero(self):
        self.syst.jacobian = np.array([[-1, 0],[0, -1]])

        sy, mat, sol = analyze_system(self.syst)
        self.assertEqual(self.syst, sy)
        self.assertIsNone(mat)
        self.assertIsNone(sol)

        sy, mat, sol = analyze_system(self.syst, filter_trivial_ss=False)
        self.assertEqual(self.syst, sy)
        self.assertIsNotNone(mat)
        self.assertIsNotNone(sol)

    def test_divergence_to_infinity(self):
        self.syst.jacobian = np.array([[1, 0],[0, 1]])

        sy, mat, sol = analyze_system(self.syst)
        self.assertEqual(self.syst, sy)
        self.assertIsNone(mat)
        self.assertIsNone(sol)

        sy, mat, sol = analyze_system(self.syst, filter_trivial_ss=False)
        self.assertEqual(self.syst, sy)
        self.assertIsNotNone(mat)
        self.assertIsNotNone(sol)

    def test_filter_masks(self):
        self.syst.jacobian = np.array([[0, 0],[0, -1]])

        sy, mat, sol = analyze_system(self.syst, filter_mask=[1])
        self.assertEqual(self.syst, sy)
        self.assertIsNotNone(mat)
        self.assertIsNotNone(sol)

class TestDataClustering(TestCase):
    def test_simple_case(self):
        """ Simple test
        """
        test_data = [(None, [2]), (None, [1])]
        res = cluster_data(test_data)

        self.assertEqual(res[0][1], [1])
        self.assertEqual(res[1][1], [2])
