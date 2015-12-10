"""
Solve stochastic differential equation
"""

import numpy as np
import numpy.random as npr


def solve_system(system, tmax=100, dt=0.1):
    """ Solve stochastic differential equation (SDE)
    """
    J = system.jacobian
    D = system.fluctuation_vector

    eq = lambda X, i: \
        sum([J[i, j]*X[j] for j in range(J.shape[1])]) \
        + np.sqrt(2 * D[i]) * npr.normal(scale=np.sqrt(dt))

    state = system.initial_state
    evolution = []

    t = 0
    while t < tmax:
        evolution.append(state)
        state = state + dt * np.array([eq(state, i) for i in range(J.shape[0])])
        t += dt

    return np.array(evolution).T

def get_steady_state(sol):
    """ Extract steady state from given solution
    """
    return sol.T[-1]
