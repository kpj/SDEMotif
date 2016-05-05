from unittest import TestCase

import numpy as np
import numpy.testing as npt

from system import SDESystem
from network_matrix import *
from nm_data_generator import *


class TestFunctions(TestCase):
    def setUp(self):
        J = np.array([[0, 0], [0, 0]])
        D_E = np.array([0, 0])
        init = np.array([1, 1])
        self.syst = SDESystem(J, D_E, D_E, init)

    def test_node_addition(self):
        # generate more systems
        res = add_node_to_system(self.syst)

        self.assertEqual(len(res), 2**4)
        for s in res:
            self.assertEqual(s.jacobian.shape, (3, 3))
            self.assertEqual(s.fluctuation_vector.shape, (3,))
            self.assertEqual(s.external_influence.shape, (3,))
            self.assertEqual(s.initial_state.shape, (3,))

    def test_data_preprocessing(self):
        def get_val(raw, enh):
            return np.sum(raw) - np.sum(enh)
        def sort_func(syst):
            return np.sum(syst.jacobian)

        syst1 = copy.deepcopy(self.syst)
        syst1.jacobian = np.array([[1,0],[0,0]])
        syst2 = copy.deepcopy(self.syst)
        syst2.jacobian = np.array([[1,0],[0,1]])

        test_data = np.array([
            [(self.syst, np.array([
                [1,2,3],
                [2,3,1],
                [3,1,2],
            ]), None), [
                (syst2, np.array([
                    [1,2,5,4],
                    [2,3,4,1],
                    [5,4,1,2],
                    [4,1,2,3]
                ]), None),
                (syst1, np.array([
                    [1,2,3,4],
                    [2,3,4,1],
                    [3,4,1,2],
                    [4,1,2,3]
                ]), None),
            ]]
        ])

        dat, xt, yt = preprocess_data(test_data, get_val, [sort_func])

        self.assertEqual(len(test_data), len(dat))
        npt.assert_array_equal(xt, [1, 2])
        self.assertEqual(yt, [0])
        npt.assert_array_equal(dat, np.array([
            [6-9, 6-11]
        ]))

    def test_example_selection(self):
        syst1 = copy.deepcopy(self.syst)
        syst1.jacobian = np.array([[1,0],[0,0]])
        syst2 = copy.deepcopy(self.syst)
        syst2.jacobian = np.array([[1,0],[0,1]])

        raw_mat = np.array([
            [1,2,3],
            [2,3,1],
            [3,1,2],
        ])
        syst1_mat = np.array([
            [1,2,3,4],
            [2,3,4,1],
            [3,4,1,2],
            [4,1,2,3]
        ])

        test_data = np.array([
            [(self.syst, raw_mat, None), [
                (syst2, np.array([
                    [1,2,5,4],
                    [2,3,4,1],
                    [5,4,1,2],
                    [4,1,2,3]
                ]), None),
                (syst1, syst1_mat, None),
            ]]
        ])
        mat = np.array([[5, 10]])

class TestColumnSorter(TestCase):
    def test_simple_case(self):
        # test data
        data = np.array([
            [-2, 3, 1, -3],
            [1, 2, 3, 4],
        ])
        sort_data = np.array([-2, 3, 1, -3])

        # sorting functions
        def sort_abs(val):
            return abs(val)
        def sort_sign(val):
            return np.sign(val)

        # test 'em
        res, rp = sort_columns(data, sort_data, [sort_abs])
        npt.assert_array_equal(res, np.array([
            [1, -2, 3, -3],
            [3, 1, 2, 4],
        ]))
        npt.assert_array_equal(rp, [2, 0, 1, 3])

        res, rp = sort_columns(data, sort_data, [sort_sign])
        npt.assert_array_equal(res, np.array([
            [-2, -3, 3, 1],
            [1, 4, 2, 3],
        ]))
        npt.assert_array_equal(rp, [0, 3, 1, 2])

        res, rp = sort_columns(data, sort_data, [sort_abs, sort_sign])
        npt.assert_array_equal(res, np.array([
            [1, -2, -3, 3],
            [3, 1, 4, 2],
        ]))
        npt.assert_array_equal(rp, [2, 0, 3, 1])

        res, rp = sort_columns(data, sort_data, [sort_sign, sort_abs])
        npt.assert_array_equal(res, np.array([
            [-2, -3, 1, 3],
            [1, 4, 3, 2],
        ]))
        npt.assert_array_equal(rp, [0, 3, 2, 1])

class TestValueFunctions(TestCase):
    def test_annihilate_low_correlations(self):
        vals = np.array([-0.5, 0.9, 0.05, 0.1, 0.2, -0.15, 0.23, -1])
        res = annihilate_low_correlations(vals, threshold=0.2)

        npt.assert_array_equal(res, np.array([-0.5, 0.9, 0, 0, 0.2, 0, 0.23, -1]))

    def test_bin_correlations(self):
        vals = np.array([-1,-0.5,-0.1,-0.02,0.04,0.08,0.6,0.99])
        res = bin_correlations(vals)

        npt.assert_array_equal(res, [-1,-1,0,0,0,0,1,1])

    def test_get_sign_changes(self):
        vals1 = np.array([1, 2,-3, 4,-5,-6, 0.05])
        vals2 = np.array([-1,2,-3,-4, 5,-6,-0.02])
        num = get_sign_changes(vals1, vals2)

        self.assertEqual(num, 3)

    def test_get_sign_changes_to_zero(self):
        vals1 = np.array([1,-0.5, 0.1,-0.12 ,0.5])
        vals2 = np.array([0, 0.5,-0.6, 0.08,-0.3])
        num = get_sign_changes(vals1, vals2)

        self.assertEqual(num, 2)

    def test_get_rank_changes(self):
        vals1 = np.array([-0.05,-0.3,-0.09,0.3,0.8])
        vals2 = np.array([-5,-6,0.02,4,0.01])
        self.assertEqual(get_rank_changes(vals1, vals2), 4)

        vals1 = np.array([0.31,0.23,0.98])
        vals2 = np.array([0.23,0.05,0.97])
        self.assertEqual(get_rank_changes(vals1, vals2), 2)

class TestSorterFunctions(TestCase):
    def setUp(self):
        self.systs = [
            SDESystem(np.array([[1,1],[2,0]]), [], [], []),
            SDESystem(np.array([[0,2],[1,0]]), [], [], []),
            SDESystem(np.array([[1,3],[4,0]]), [], [], []),
            SDESystem(np.array([[1,4],[3,1]]), [], [], [])
        ]

    def test_sort_by_network_density(self):
        res = sorted(self.systs, key=sort_by_network_density)
        self.assertEqual(res,
            [self.systs[1], self.systs[0], self.systs[2], self.systs[3]])

    def test_sort_by_indeg(self):
        res = sorted(self.systs, key=sort_by_indeg)
        self.assertEqual(res,
            [self.systs[0], self.systs[1], self.systs[2], self.systs[3]])

    def test_sort_by_outdeg(self):
        res = sorted(self.systs, key=sort_by_outdeg)
        print([f.jacobian for f in res])
        self.assertEqual(res,
            [self.systs[1], self.systs[0], self.systs[3], self.systs[2]])

    def test_sort_by_cycle_num(self):
        res = sorted(self.systs, key=sort_by_cycle_num)
        print([f.jacobian for f in res])
        self.assertEqual(res,
            [self.systs[1], self.systs[0], self.systs[2], self.systs[3]])

class TestClustering(TestCase):
    def test_hamming(self):
        mat = np.array([
            [8,1,1,3,3],
            [3,1,1,3,3],
            [5,1,2,3,5]
        ])

        res, _ = cluster_data(mat, 'hamming')
        npt.assert_array_equal(res, np.array([
            [1, 1, 3, 8, 3],
            [1, 1, 3, 3, 3],
            [1, 2, 3, 5, 5]
        ]))
