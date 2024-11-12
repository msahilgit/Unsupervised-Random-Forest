import numpy as np
import sys
sys.path.append('../../0_python_modules/')
import extras
import msm_analysis as ana




cv=range(5)
arcs = [2, 3, 4, 6]
lags = np.arange(5, 51, 5)
states = [3, 4, 5, 6]


load_labels = [True for _ in lags]
labels = []
weights = []

ignores = np.array([
            [1, 6, 5, 6],
            [2, 2, 10, 4],
            [3, 6, 10, 6],
            [4, 6, 10, 6]
            ])

for i in cv:

    for s in states:
        
        ids = np.zeros(( len(arcs), len(lags), s, 2 )) + np.nan
        ginis = np.zeros(( len(arcs), len(lags), s )) + np.nan

        for a,arc in enumerate(arcs):

            for b,lag in enumerate(lags):

                if load_labels[b]:
                    l = np.load('../1_data/labels.npz')
                    l = np.concatenate(([l[d][:-lag] for d in list(l)]))
                    labels.append(l)
                    weights.append( np.array([np.where(l==k)[0].shape[0]/l.shape[0] for k in np.unique(l)]) )
                    load_labels[b] = False

                proceed = True
                if np.any( np.all( [i, arc, lag, s] == ignores, axis=1) ):
                    proceed = False


                if proceed:

                    mdtrj = np.load(f'saved_dtrjs/dtrj_s{s}_f{arc}_{lag}_{i}.npy')

                    pis, ids[a,b] = ana.get_pis(mdtrj, labels[b], ids=True) 

                    pis = ana.get_reweighted_pis(pis, weights[b])
                    ginis[a,b] = ana.get_gini(pis)

                print(i,s,arc,lag)

        np.save(f'saved_analyze_states/ids_s{s}_{i}.npy', ids)
        np.save(f'saved_analyze_states/ginis_s{s}_{i}.npy', ginis)





