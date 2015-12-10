"""
Setup system
"""

import numpy as np


class SDESystem(object):
    """ Bundle import SDE information
    """
    def __init__(self, J, D, I):
        self.jacobian = J
        self.fluctuation_vector = D
        self.initial_state = I

def generate_system():
    """ Generate system according to paper
    """
    k_12 = k_13 = k_out = 1
    k_23 = 2
    D = 1

    jacobian = np.array([
        [-(k_12 + k_12),    0,      0],
        [k_12,              -k_23,  0],
        [k_13,              k_23,   -k_out]
    ])
    fluctuation_vector = np.array([D, 0, 0])
    initial_state = np.array([2.5, 1.25, 5.])

    system = SDESystem(jacobian, fluctuation_vector, initial_state)
    return system
