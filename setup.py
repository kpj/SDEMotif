"""
Setup system
"""

import numpy as np
import numpy.random as npr


class SDESystem(object):
    """ Bundle import SDE information
    """
    def __init__(self, J, D, E, I):
        self.jacobian = J
        self.fluctuation_vector = D
        self.external_influence = E
        self.initial_state = I

def generate_basic_system(v_in=5, k_m=1, k_23=2, D=1):
    """ Generate system according to paper
    """
    k_12 = k_13 = k_out = k_m

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

def generate_random_plus(num=5):
    """ Generate randomized versions of initial system plus one node
    """
    res = [generate_basic_system()]
    for _ in range(num-1):
        s = generate_basic_system()

        # extend by fourth node
        s.fluctuation_vector = np.append(s.fluctuation_vector, 0)
        s.external_influence = np.append(s.external_influence, 0)
        s.initial_state = np.append(s.initial_state, 0)

        s.jacobian = npr.choice([0, 1], (4, 4))
        res.append(s)
    return res

def generate_varied_parameters(num=5):
    """ Vary parameters of initial system
    """
    res = [generate_basic_system()]
    for _ in range(num-1):
        s = generate_basic_system(
            k_m=npr.uniform(0, 3),
            k_23=npr.uniform(0, 3))
        res.append(s)
    return res


generate_systems = generate_varied_parameters
