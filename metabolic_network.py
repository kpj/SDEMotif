import cobra
from cameo.network_analysis.networkx_based import model_to_network

from reaction_finder import detect_ffl


def main():
    # load model
    model = cobra.io.read_sbml_model('data/HMRdatabase2_00_Cobra.xml')
    graph = model_to_network(model)

    # extract motifs
    motifs = []
    for cs in detect_ffl(graph):
        if None in cs: continue
        if cs in motifs: continue
        motifs.append(cs)
    motifs_n = [tuple(map(lambda x: x.name, mot)) for mot in motifs]

    # output sensible ones
    marker = [
        'cofactors', 'pool',
        'Na', 'Pi', 'CoA', 'apo',
        'H2O', 'O2_O2', 'CO2_CO2', 'H+_H',
        'AMP', 'ADP', 'ATP', 'UMP', 'UDP', 'UTP', 'GMP', 'GDP', 'GTP',
        'NAD', 'NAD'
    ]
    match = lambda x: all(not w in x for w in marker)

    with open('results/metabolite_list.txt', 'w') as fd:
        for m in motifs_n:
            if not all(map(match, m)): continue
            fd.write(f'{m}\n')

if __name__ == '__main__':
    main()
