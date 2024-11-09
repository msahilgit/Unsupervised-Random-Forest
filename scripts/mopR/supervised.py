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

sys.path.append('../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


traj_data = np.load('../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data)]
dists = np.concatenate(( np.load('../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))

#features = np.concatenate(( traj_data ))[::20]
#labels = dists[::20]
randoms = np.loadtxt('../../1_datasets/randoms.txt', dtype=int)


#accu = np.zeros((len(randoms)))
for i in range(len(randoms)):
    #xtrain, xtest, ytrain, ytest = splt(features, labels, test_size=0.3, random_state=randoms[i])
    #reg_ = rfr(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    #reg_.fit(xtrain, ytrain)

    #pred_ = reg_.predict(xtest)
    #accu[i] = mtr.mean_squared_error(ytest,pred_)


    #xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
    #pkl.dump(reg_, open(f'saved_supervised/reg_supervised{i}.pkl','wb'))

    #fimp = reg_.feature_importances_
    #np.save(f'saved_supervised/fimp_supervised_{i}.npy', fimp)
    fimp = np.load(f'saved_supervised/fimp_supervised_{i}.npy')
    fimp = fimp.argsort()[::-1][:40]
    tic2 = [i[:,fimp] for i in traj_data]
    tic2 = np.concatenate(( coor.tica( tic2, lag=10, dim=2).get_output() ))
    #hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1)
    #np.save(f'saved_supervised/hist_{i}.npy', hh[4])
    #np.save(f'saved_supervised/extents_{i}.npy', hh[:4])

    #hh = nh.hist_range(tic2, dists, mini=0, maxi=0.35)
    #np.save(f'saved_supervised/different_state_bound/hist_{i}.npy', hh[4])
    #np.save(f'saved_supervised/different_state_bound/extents_{i}.npy', hh[:4])

    hh = nh.hist_range(tic2, dists, mini=1, maxi=10)
    np.save(f'saved_supervised/different_state_unbound/hist_{i}.npy', hh[4])
    np.save(f'saved_supervised/different_state_unbound/extents_{i}.npy', hh[:4])

#np.save('saved_supervised/accu_supervised.npy', accu)


