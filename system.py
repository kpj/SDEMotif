"""
General system class
"""

import numpy as np


class SDESystem(object):
    """ Bundle import SDE information
    """
    def __init__(self, J, D, E, I):
        self.jacobian = J
        self.fluctuation_vector = np.asarray(D)
        self.external_influence = np.asarray(E)
        self.initial_state = np.asarray(I)
