"""
Setup system
"""

import numpy as np


class SDESystem(object):
    """ Bundle import SDE information
    """
    def __init__(self, J, D, E, I):
        self.jacobian = J
        self.fluctuation_vector = D
        self.external_influence = E
        self.initial_state = I

def generate_system():
    """ Generate system according to paper
    """
    v_in = 5
    k_12 = k_13 = k_out = 1
    k_23 = 2
    D = 1

    jacobian = np.array([
        [-(k_12 + k_12),    0,      0],
        [k_12,              -k_23,  0],
        [k_13,              k_23,   -k_out]
    ])
    external_influence = np.array([v_in, 0, 0])
    fluctuation_vector = np.array([D, 0, 0])
    initial_state = np.array([0, 0, 0])

    system = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)
    return system
