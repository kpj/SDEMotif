from unittest import TestCase

import numpy as np
import numpy.testing as npt

from setup import *


class TestSystemFromString(TestCase):
    def test_simple_case(self):
        string = '1 2 ; 3 4'
        syst = system_from_string(string)

        npt.assert_array_equal(syst.jacobian, np.array([
            [1, 2],
            [3, 4]
        ]))
