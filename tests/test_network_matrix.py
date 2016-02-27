from unittest import TestCase

import numpy as np
import numpy.testing as npt

from system import SDESystem
from network_matrix import *


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

    def test_column_sorting(self):
        # test data
        data = np.array([
            [-2, 2, 1, -1],
            [1, 2, 3, 4],
        ])
        sort_data = np.array([-2, 2, 1, -1])

        # sorting functions
        def sort_abs(val):
            return abs(val)
        def sort_sign(val):
            return np.sign(val)

        # test 'em
        res = sort_columns(data, sort_data, [sort_abs, sort_sign])
        npt.assert_array_equal(res, np.array([
            [-1, 1, -2, 2],
            [4, 3, 1, 2]
        ]))

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
        self.assertEqual(xt, [1, 2])
        self.assertEqual(yt, [0])
        npt.assert_array_equal(dat, np.array([
            [6-9, 6-11]
        ]))
