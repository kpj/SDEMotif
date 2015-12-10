"""
Reproduce figure 6 from Steuer et al., (2003)
"""

import numpy as np

from setup import generate_system
from solver import solve_system, get_steady_state
from plotter import plot_system_evolution, plot_ss_scatter


def main():
    """ Main interface
    """
    system = generate_system()

    ss_data = []
    for _ in range(100):
        sol = solve_system(system)
        #plot_system_evolution(sol)

        ss = get_steady_state(sol)
        ss_data.append(ss)

    plot_ss_scatter(np.array(ss_data))


if __name__ == '__main__':
    main()
