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
import cluster as cls


features = np.load('../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]

permute_pmt = np.load('saved_furf/permute_pmt_mean.npy')
nobs = features.shape[0]


accu = np.zeros((9,len(randoms),2))

for h in range(2,11):

    _, klabels = cls.kmed_pmt( nobs, permute_pmt, nclus=h, init='random', max_iter=5000, random_seed=h )
    np.save(f'saved_kmed_pmt/try_2/klabels_{h}.npy', klabels)

    for i in range(len(randoms)):
        xtrain, xtest, ytrain, ytest = splt(features, klabels, test_size=0.3, random_state=randoms[i])
        clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
        clf_.fit(xtrain, ytrain)

        pred_ = clf_.predict(xtest)
        accu[h-2,i,0] = acc(ytest,pred_)
        accu[h-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )

        xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
        pkl.dump(clf_, open(f'saved_kmed_pmt/try_2/clf_klabel{h}_{i}.pkl','wb'))
        fimp = clf_.feature_importances_
        np.save(f'saved_kmed_pmt/try_2/fimp_klabel{h}_{i}.npy', fimp)

        top200 = fimp.argsort()[::-1][:200]
        tic2 = [traj_data[i][:,top200] for i in range(6)]
        tic2 = coor.tica(tic2, lag=700, dim=2).get_output()
        tic2 = [tic2[0], tic2[4], tic2[3]]
        hist_ = []
        extents = np.zeros((len(dists), 4))
        for k in range(len(dists)):
            hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
            hist_.append(hh[4])
            extents[k] = hh[:4]
        np.save(f'saved_kmed_pmt/try_2/hist_{h}_{i}.npy', hist_)



np.save('saved_kmed_pmt/try_2/accu_kmed.npy', accu)


