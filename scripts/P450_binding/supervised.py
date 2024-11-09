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


traj_data = [ np.load(f'../1_datasets/p450_2018/features{i}.npy') for i in range(3) ]
dists = [ np.loadtxt(f'../1_datasets/p450_2018/distance{i}.xvg', comments=['@','#'])[:,1] for i in range(3) ]

labels = np.concatenate(( dists ))[::10]
labels = (labels <= 0.6) * 1


features = np.concatenate(( traj_data ))[::10]
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)


accu = np.zeros((len(randoms),2))

for i in range(len(randoms)):
    xtrain, xtest, ytrain, ytest = splt(features, labels, test_size=0.3, random_state=randoms[i])
    clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    clf_.fit(xtrain, ytrain)

    pred_ = clf_.predict(xtest)
    accu[i,0] = acc(ytest,pred_)
    accu[i,1] = extras.get_f1_score( cfm(ytest, pred_) )

    xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
    pkl.dump(clf_, open(f'saved_supervised/clf_{i}.pkl','wb'))
    fimp = clf_.feature_importances_
    np.save(f'saved_supervised/fimp_{i}.npy', fimp)

    top200 = fimp.argsort()[::-1][:200]
    tic2 = [traj_data[i][:,top200] for i in range(len(dists))]
    tic2 = coor.tica(tic2, lag=1000, dim=2).get_output()
    hist_ = []
    extents = np.zeros((len(dists), 4))
    for k in range(len(dists)):
        hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
        hist_.append(hh[4])
        extents[k] = hh[:4]
    np.save(f'saved_supervised/hist_{i}.npy', hist_)
    np.save(f'saved_supervised/extents_{i}.npy', extents)



np.save('saved_supervised/accu_supervised.npy', accu)


