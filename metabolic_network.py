import itertools

import cobra
from cameo.network_analysis.networkx_based import model_to_network

from reaction_finder import detect_ffl


def reaction_ids_unique(cs, graph):
    """ Check that unique reactions are used for FFL
    """
    d1 = graph.get_edge_data(cs[0], cs[1])
    d2 = graph.get_edge_data(cs[0], cs[2])
    d3 = graph.get_edge_data(cs[1], cs[2])
    assert graph.get_edge_data(cs[1], cs[0]) is None
    assert graph.get_edge_data(cs[2], cs[0]) is None
    assert graph.get_edge_data(cs[2], cs[1]) is None

    for r_group in itertools.product(d1.values(), d2.values(), d3.values()):
        idx = tuple(map(lambda x: x['reaction'].id, r_group))
        if len(set(idx)) < 3:
            return False
    return True

def main():
    # load model
    model = cobra.io.read_sbml_model('data/HMRdatabase2_00_Cobra.xml')
    graph = model_to_network(model)

    # extract motifs
    motifs = []
    for cs in detect_ffl(graph):
        if None in cs or cs in motifs:
            continue
        if not reaction_ids_unique(cs, graph):
            continue

        motifs.append(cs)
    motifs_n = [tuple(map(lambda x: x.name, mot)) for mot in motifs]

    # output sensible ones
    marker = [
        'cofactors', 'pool',
        'Na', 'Pi', 'CoA', 'apo',
        'H2O', 'O2_O2', 'CO2_CO2', 'H+_H', 'HCO3-',
        'AMP', 'ADP', 'ATP', 'UMP', 'UDP', 'UTP', 'GMP', 'GDP', 'GTP',
        'NAD', 'NAD'
    ]
    match = lambda x: all(w not in x for w in marker)

    with open('results/metabolite_list.txt', 'w') as fd:
        for m in motifs_n:
            if not all(map(match, m)):
                continue
            fd.write(f'{m}\n')


if __name__ == '__main__':
    main()
