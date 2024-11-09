import numpy as np
np.bool = np.bool_
import copy as cp
import sys
import matplotlib as mt
import matplotlib.pyplot as plt
import pickle as pkl
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as splt
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
import sklearn.metrics as mtr
import scipy as sc
import scipy.cluster as scc
from tqdm import tqdm
import time
import pyemma
import pyemma.coordinates as coor
import pyemma.plots as mplt

sys.path.append('../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


traj_data = np.load('../1_datasets/PASH1/prconf_dists.npy')

randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

create_permute = syn.synthetic_data(traj_data[0], size=1)
create_permute.permute()
pfeatures, plabels = create_permute.get_output()

np.save('saved_furf/pfeatures.npy', pfeatures)
np.save('saved_furf/plabels.npy', plabels)

clfs = []
accu = np.zeros((len(randoms),2))
for i in range(len(randoms)):
    xtrain, xtest, ytrain, ytest = splt(pfeatures, plabels, test_size=0.3, random_state=randoms[i])
    clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    clf_.fit(xtrain, ytrain)

    pred_ = clf_.predict(xtest)
    accu[i,0] = acc(ytest,pred_)
    accu[i,1] = extras.get_f1_score( cfm(ytest, pred_) )


    xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
    clfs.append(clf_)
    pkl.dump(clf_, open(f'saved_furf/clf_permute{i}.pkl','wb'))

np.save('saved_furf/accu_permute.npy', accu)


permute_pmt = pmt.calculate_condenced_pmt_multi( clfs, traj_data[0], n_jobs=94 )
np.save('saved_furf/permute_pmt_mean.npy', permute_pmt)


hc_permute = scc.hierarchy.linkage(permute_pmt, method='average')
np.save('saved_furf/hc_permute.npy', hc_permute)




accu = np.zeros((9,len(randoms),2))
for h in range(2,11):
    hlabels_ = extras.get_hc_dtraj(hc_permute, nids=h)

    for i in range(len(randoms)):
        xtrain, xtest, ytrain, ytest = splt(traj_data[0], hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
        clf_.fit(xtrain, ytrain)

        pred_ = clf_.predict(xtest)
        accu[h-2,i,0] = acc(ytest,pred_)
        accu[h-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )

        xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
        pkl.dump(clf_, open(f'saved_furf/clf_hlabel{h}_{i}.pkl','wb'))
        fimp = clf_.feature_importances_
        np.save(f'saved_furf/fimp_hlabel{h}_{i}.npy', fimp)


np.save('saved_furf/accu_hc.npy', accu)



