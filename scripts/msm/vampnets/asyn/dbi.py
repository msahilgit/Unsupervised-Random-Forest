import numpy as np
import sklearn as skl
import sklearn.metrics as mtr
import sys
sys.path.append('../../0_python_modules/')
import extras
import msm_analysis as ana
import idp_analysis as idp


features = np.load('../1_data/ca_dists.npy')[0]
features = (features-np.min(features, axis=0)) / ( np.max(features, axis=0)-np.min(features,axis=0) )
randoms = np.loadtxt('../1_data/randoms.txt', dtype=int)


cv=range(5)
arcs = [2, 3, 4, 6]
lags = np.arange(5, 71, 5)
states = [2, 3, 4, 5, 6]

dbis = np.zeros(( len(cv), len(arcs), len(lags), len(states) )) + np.nan
for i in cv:
    for a,arc in enumerate(arcs):
        for b,l in enumerate(lags):
            for c,s in enumerate(states):

                mdtrj = np.load(f'saved_dtrjs/dtrj_s{s}_f{arc}_{l}_{i}.npy')

                dbis[i,a,b,c] = mtr.davies_bouldin_score(features[:-l], mdtrj) 


np.save('saved_dbi/dbis.npy', dbis)




