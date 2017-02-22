"""
General system class
"""

import pickle

import numpy as np


class SDESystem(object):
    """ Bundle import SDE information
    """
    def __init__(self, J, D, E, I):
        self.jacobian = J
        self.fluctuation_vector = np.asarray(D)
        self.external_influence = np.asarray(E)
        self.initial_state = np.asarray(I)

    def save(self, fname):
        with open(fname, 'wb') as fd:
            pickle.dump(self, fd)

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as fd:
            return pickle.load(fd)
