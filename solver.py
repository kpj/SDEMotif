"""
Solve stochastic differential equation
"""

import numpy as np
import numpy.random as npr


def solve_system(system, tmax=100, dt=0.1, seed=None):
    """ Solve stochastic differential equation (SDE)
    """
    J = system.jacobian
    D = system.fluctuation_vector
    E = system.external_influence
    dim = J.shape[0]

    state = system.initial_state
    evolution = []
    dtsq = np.sqrt(dt)
    tdsq = np.sqrt(2*D)

    np.seterr(all='raise')
    npr.seed(seed)

    t = 0
    while t < tmax:
        evolution.append(state)

        delta = J.dot(state) + E
        fluc = tdsq * dtsq * npr.normal(size=dim)
        state = state + dt * delta + fluc

        t += dt

    return np.array(evolution).T
