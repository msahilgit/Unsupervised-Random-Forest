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


features = np.load('../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]


clfs = []
for i in range(len(randoms)):
    clfs.append( pkl.load(open(f'saved_furf/clf_permute{i}.pkl','rb')) )


data_sizes = np.arange( 0.1, 1.1, 0.1 )
nobs = features.shape[0]

np.random.seed(randoms[0])
rorder = np.random.permutation(nobs)

for s in data_sizes:

    tsize = int( nobs * s )
    tfeatures = features[ rorder[:tsize] ]
    iters=int(s*10)

    permute_pmt = pmt.calculate_condenced_pmt_multi( clfs, tfeatures, n_jobs=94 )
    np.save(f'saved_data_size_pmt/permute_pmt_mean_{iters}.npy', permute_pmt)


    hc_permute = scc.hierarchy.linkage(permute_pmt, method='average')
    np.save(f'saved_data_size_pmt/hc_permute_{iters}.npy', hc_permute)


    accu = np.zeros((9,len(randoms),2))
    for h in range(2,11):
        hlabels_ = extras.get_hc_dtraj(hc_permute, nids=h)

        for i in range(len(randoms)):
            xtrain, xtest, ytrain, ytest = splt(tfeatures, hlabels_, test_size=0.3, random_state=randoms[i])
            clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
            clf_.fit(xtrain, ytrain)

            pred_ = clf_.predict(xtest)
            accu[h-2,i,0] = acc(ytest,pred_)
            accu[h-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )

            xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
            pkl.dump(clf_, open(f'saved_data_size_pmt/clf_s{iters}_hlabel{h}_{i}.pkl','wb'))
            fimp = clf_.feature_importances_
            np.save(f'saved_data_size_pmt/fimp_s{iters}_hlabel{h}_{i}.npy', fimp)

            top200 = fimp.argsort()[::-1][:200]
            tic2 = [traj_data[i][:,top200] for i in range(6)]
            tic2 = coor.tica(tic2, lag=100, dim=2).get_output()
            tic2 = [tic2[0], tic2[4], tic2[3]]
            hist_ = []
            extents = np.zeros((len(dists), 4))
            for k in range(len(dists)):
                hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
                hist_.append(hh[4])
                extents[k] = hh[:4]
            np.save(f'saved_data_size_pmt/hist_s{iters}_{h}_{i}.npy', hist_)
            np.save(f'saved_data_size_pmt/extents_s{iters}_{h}_{i}.npy', extents)


    np.save(f'saved_data_size_pmt/accu_hc_s{iters}.npy', accu)


