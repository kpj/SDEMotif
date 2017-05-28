"""
Generate and save system configurations.
All generator functions must start with `generate_`
"""

import sys
import typing
import itertools

import numpy as np
import numpy.random as npr

from tqdm import tqdm

from system import SDESystem
from utils import cache_data


def generate_basic_system(v_in=5, k_m=1, k_23=2, D=.1):
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
    initial_state = np.array([1, 1, 1])

    system = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)
    return system

def generate_motifs():
    """ Generate all motifs from some paper.
        However return function to generate motif with given parameters
    """
    v_in = 5
    D = .1

    def gen(jacobian):
        initial_state = np.array([1, 1, 1])

        return [
            lambda v_in=5, k_m=1, k_23=2, D=.1:
                SDESystem(
                    jacobian(k_m, k_23), [D, 0, 0],
                    [v_in, 0, 0], initial_state),
            lambda v_in=5, k_m=1, k_23=2, D=.1:
                SDESystem(
                    jacobian(k_m, k_23), [0, D, 0],
                    [0, v_in, 0], initial_state),
            lambda v_in=5, k_m=1, k_23=2, D=.1:
                SDESystem(
                    jacobian(k_m, k_23), [0, 0, D],
                    [0, 0, v_in], initial_state)
        ]

    jacobians = [
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      0],
            [k_m,            -k_23,  0],
            [k_m,            0,      -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      0],
            [k_m,            -k_23,  k_23],
            [0,              0,      -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      0],
            [k_m,            -k_23,  0],
            [0,              k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      0],
            [0,              -k_23,  k_23],
            [k_m,            k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      k_m],
            [0,              -k_23,  k_23],
            [0,              k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      k_m],
            [0,              -k_23,  k_23],
            [k_m,            k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      0],
            [k_m,            -k_23,  0],
            [k_m,            k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      k_m],
            [k_m,            -k_23,  0],
            [0,              k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      0],
            [k_m,            -k_23,  k_23],
            [k_m,            k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      k_m],
            [k_m,            -k_23,  k_23],
            [k_m,            0,      -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      k_m],
            [k_m,            -k_23,  0],
            [k_m,            k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   0,      k_m],
            [k_m,            -k_23,  k_23],
            [k_m,            k_23,   -k_m]
        ]),
        lambda k_m, k_23: np.array([
            [-(k_m + k_m),   k_m,    k_m],
            [k_m,            -k_23,  k_23],
            [k_m,            k_23,   -k_m]
        ]),
    ]

    return [gen(jac) for jac in jacobians]

def generate_two_node_system(v_in=5, D=.1, k_m=.5, k_23=.5):
    """ Generate system with only two nodes
    """
    jacobian = np.array([
        [-1,    0],
        [k_m+k_23,  -1]
    ])
    external_influence = np.array([v_in, 0])
    fluctuation_vector = np.array([D, 0])
    initial_state = np.array([1, 1])

    system = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)
    return system

def generate_v_out(v_in=5, k_m=1, k_23=2, D=.1):
    """ Generate V-motif
    """
    k_12 = k_13 = k_out = k_m

    jacobian = np.array([
        [-(k_12 + k_12),    0,       0],
        [k_12,              -k_out,  0],
        [k_13,              0,       -k_out]
    ])
    external_influence = np.array([v_in, 0, 0])
    fluctuation_vector = np.array([D, 0, 0])
    initial_state = np.array([1, 1, 1])

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
        s.initial_state = np.append(s.initial_state, 1)

        s.jacobian = npr.choice([0, 1], (4, 4))
        res.append(s)
    return res

def generate_plus():
    """ Generate initial system plus one node
    """
    def with_fourth_node(hs, vs):
        s = generate_basic_system()

        # adjust other vectors
        s.fluctuation_vector = np.append(s.fluctuation_vector, 0)
        s.external_influence = np.append(s.external_influence, 0)
        s.initial_state = np.append(s.initial_state, 1)

        # check validity of input
        dim = s.jacobian.shape[0]
        assert len(hs) == dim, 'Horizontal stack must have length %d' % dim
        assert len(vs) == dim+1, 'Vertical stack must have length %d' % (dim+1)

        # adjust jacobian
        s.jacobian = np.hstack((s.jacobian, np.array(hs).reshape(-1, 1)))
        s.jacobian = np.vstack((s.jacobian, vs))

        return s

    res = [generate_basic_system()]
    res.append(with_fourth_node(
        [0, 0, 0], [0, 0, 0, -2]))

    for conn in itertools.product([0, 1], repeat=3):
        cur = list(conn) + [-2]
        for foo in [[1,0,0], [0,1,0], [0,0,1]]:
            res.append(with_fourth_node(
                foo, cur))

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

def generate_all(size=4, force_self_inhibition=False):
    """ Generate all networks of given size
    """
    assert size > 0, 'Require positive network size'

    # get all possible edges as node pairs
    all_edges = list(itertools.product(range(size), repeat=2))

    # compute all possible subsets
    def powerset(inp):
        return itertools.chain.from_iterable(
            itertools.combinations(inp, r) for r in range(len(inp)+1))

    all_subgraphs = powerset(all_edges)

    # generate system definitions
    v_in = 5
    D = 1
    k = 1

    res = []
    with tqdm(total=2**len(all_edges)) as pbar:
        for graph in all_subgraphs:
            # input only on first node
            external_influence = np.array([v_in] + [0] * (size-1))
            fluctuation_vector = np.array([D] + [D] * (size-1))
            initial_state = np.array([1] * size)

            # create fitting jacobian
            jacobian = np.zeros((size, size))
            for i, j in graph:
                if i == j:
                    jacobian[i, j] = -k
                else:
                    jacobian[i, j] = k

            if force_self_inhibition:
                if not (np.diagonal(jacobian) < 0).all():
                    continue

            # assemble system
            system = SDESystem(
                jacobian, fluctuation_vector,
                external_influence, initial_state)

            res.append(system)
            pbar.update()

    return res

def generate_steadystate_and_divergence():
    """ Contrast correlation of SS and diverging case
    """
    external_influence = np.array([1, 0, 0])
    fluctuation_vector = np.array([1, 0, 0])
    initial_state = np.array([1, 1, 1])
    jacobian = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0]
    ])

    # diverging case
    d_sys = SDESystem(
        jacobian, fluctuation_vector,
        external_influence, initial_state)

    # steady state
    selfinh_jacobian = jacobian.copy()
    np.fill_diagonal(selfinh_jacobian, -1)
    ss_sys = SDESystem(
        selfinh_jacobian, fluctuation_vector,
        external_influence, initial_state)

    return [d_sys, ss_sys]

def generate_cycles(size=10):
    """ Generate one cycle and start adding interfering edges
    """
    def get_system(jacobian):
        """ Create proper system from given Jacobian
        """
        external_influence = np.array([5] + [0] * (size-1))
        fluctuation_vector = np.full((size,), 0)
        fluctuation_vector[0] = 1
        initial_state = np.full((size,), 1)

        return SDESystem(
            jacobian, fluctuation_vector,
            external_influence, initial_state)

    # generate cycle
    mat = np.zeros((size, size))
    np.fill_diagonal(mat, -1)
    ind = np.arange(size)
    mat[ind, ind-1] = 1

    systs = [get_system(mat)]

    # generate variants
    foo = np.copy(mat)
    foo[0, 1] = -1
    systs.append(get_system(foo))

    foo = np.copy(mat)
    foo[0, 3] = -1
    systs.append(get_system(foo))

    foo = np.copy(mat)
    foo[0, 5] = -1
    systs.append(get_system(foo))

    foo = np.copy(mat)
    foo[0, 7] = -1
    systs.append(get_system(foo))

    foo = np.copy(mat)
    foo[0, 9] = -1
    systs.append(get_system(foo))

    return systs

def system_from_string(string):
    """
    System string:
        string := jacobian

    Input + fluctuations will be assumed to happen on node 0
    """
    mat = []
    for row in string.split(';'):
        mat.append(np.fromstring(row, sep=' '))
    J = np.array(mat)

    E = np.array([5] + [0] * (J.shape[0]-1))
    F = np.array([1] + [0] * (J.shape[0]-1))
    I = np.array([1] * J.shape[0])

    return SDESystem(J, F, E, I)


def load_systems(fname):
    """ Support for loading pickled systems from given file
    """
    return np.load(fname)

if __name__ == '__main__':
    if not len(sys.argv) in (1, 2):
        print('Usage: %s [function selector]' % sys.argv[0])
        sys.exit(1)

    print('Choose generator:')
    keys = sorted(globals().keys())
    for i, name in enumerate(keys):
        if name.startswith('generate_') and isinstance(globals()[name], typing.Callable):
            print(' [%d] - %s' % (i, name))

    if len(sys.argv) == 1:
        choice = int(input('-> '))
    elif len(sys.argv) == 2:
        choice = int(sys.argv[1])
        print('-> %d' % choice)
    else:
        raise RuntimeError('This should never happen :-)')
    func = globals()[keys[choice]]

    res = func()
    cache_data(res, fname='results/systems')
