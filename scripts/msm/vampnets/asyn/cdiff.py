import numpy as np
import sklearn as skl
import sklearn.metrics as mtr
import sys
sys.path.append('../../0_python_modules/')
import extras
import msm_analysis as ana
import idp_analysis as idp


features = np.load('../1_data/ca_dists.npy')[0]
randoms = np.loadtxt('../1_data/randoms.txt', dtype=int)


cv=range(5)
arcs = [2, 3, 4, 6]
lags = np.arange(5, 71, 5)
states = [2, 3, 4, 5, 6]

cmean_strict = np.zeros(( len(cv), len(arcs), len(lags), len(states) )) + np.nan
cmean_loose = np.zeros(( len(cv), len(arcs), len(lags), len(states) )) + np.nan
cmedian_strict = np.zeros(( len(cv), len(arcs), len(lags), len(states) )) + np.nan
cmedian_loose = np.zeros(( len(cv), len(arcs), len(lags), len(states) )) + np.nan

for i in cv:
    for a,arc in enumerate(arcs):
        for b,l in enumerate(lags):
            for c,s in enumerate(states):

                mdtrj = np.load(f'saved_dtrjs/dtrj_s{s}_f{arc}_{l}_{i}.npy')

                cmean_loose[i,a,b,c], cmean_strict[i,a,b,c], cmedian_loose[i,a,b,c], cmedian_strict[i,a,b,c] = idp.get_contact_diffs(mdtrj, features, cutoff=0.5)




np.save('saved_cdiff/cmean_strict.npy', cmean_strict)
np.save('saved_cdiff/cmean_loose.npy', cmean_loose)
np.save('saved_cdiff/cmedian_strict.npy', cmedian_strict)
np.save('saved_cdiff/cmedian_loose.npy', cmedian_loose)




