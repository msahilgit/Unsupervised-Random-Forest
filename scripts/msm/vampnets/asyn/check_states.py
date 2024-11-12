import numpy as np
import random
import os




cv=range(5)
arcs = [2, 3, 4, 6]
lags = np.arange(5, 71, 5)
states = [2, 3, 4, 5, 6]

bad=0
for i in cv:
    for arc in arcs:
        for l in lags:
            for s in states:

                mdtrj = np.load(f'saved_dtrjs/dtrj_s{s}_f{arc}_{l}_{i}.npy')
                ns = np.unique(mdtrj).shape[0]
                if ns < s:
                    print(i, arc, l, s, ns)
                    bad += 1


print(f'\n {bad} bad dtrjs were found \n')





