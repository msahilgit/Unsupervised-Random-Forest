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

sys.path.append('../../../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


traj_data = np.load('../../../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data)]
dists = np.concatenate(( np.load('../../../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))

features = np.concatenate(( traj_data ))[::20]
labels = dists[::20]
randoms = np.loadtxt('../../../../1_datasets/randoms.txt', dtype=int)

fval = 0

indices = np.arange(features.shape[0])
nofs = features.shape[1]
accu_train = np.zeros(( len(randoms), nofs ))
accu_test = np.zeros(( len(randoms), nofs ))

for i in range(len(randoms)):

    train, test = splt( indices, test_size=0.3, random_state=randoms[i])

    xtrain = features[train]
    ytrain = labels[train]

    reg = pkl.load(open(f'../../saved_supervised/reg_supervised{i}.pkl','rb'))

    for f in range(nofs):

        xtrain = features[train]
        xtrain[:,f] = fval
        ytrain = labels[train]
        pred_train = reg.predict(xtrain)

        accu_train[i,f] = mtr.mean_squared_error(ytrain, pred_train)

        xtest = features[test]
        xtest[:,f] = fval
        ytest = labels[test]
        pred_test = reg.predict(xtest)

        accu_test[i,f] = mtr.mean_squared_error(ytest, pred_test)

        print(i,f)


np.save(f'saved_predict/accu_train_{fval}.npy', accu_train)
np.save(f'saved_predict/accu_test_{fval}.npy', accu_test)


