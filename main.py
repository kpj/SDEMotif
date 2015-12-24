"""
Reproduce figure 6 from Steuer et al., (2003)
"""

import numpy as np

from setup import generate_systems
from solver import solve_system, get_steady_state
from plotter import plot_system_overview


def get_steady_states(system, repetition_num=100):
    """ Generate steady states for given system
    """
    ss_data = []
    for _ in range(repetition_num):
        sol = solve_system(system)

        ss = get_steady_state(sol)
        ss_data.append(ss)
    return np.array(ss_data), sol

def main():
    """ Main interface
    """
    systems = generate_systems()

    data = []
    for syst in systems:
        ss_data, sol = get_steady_states(syst)
        data.append((syst, ss_data, sol))

    plot_system_overview(data)


if __name__ == '__main__':
    main()
