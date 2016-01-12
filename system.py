"""
General system class
"""

class SDESystem(object):
    """ Bundle import SDE information
    """
    def __init__(self, J, D, E, I):
        self.jacobian = J
        self.fluctuation_vector = D
        self.external_influence = E
        self.initial_state = I
