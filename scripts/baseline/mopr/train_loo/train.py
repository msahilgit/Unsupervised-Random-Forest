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

nofs = features.shape[1]

accu = np.zeros(( nofs, len(randoms) ))

for f in range(nofs):
    sfeatures = np.delete( features, f, axis=1 )

    for i in range(len(randoms)):
        xtrain, xtest, ytrain, ytest = splt(sfeatures, labels, test_size=0.3, random_state=randoms[i])
        reg = rfr(n_estimators=1000, random_state=randoms[i], n_jobs=94)
        reg.fit(xtrain, ytrain)

        pred = reg.predict(xtest)
        accu[f,i] = mtr.mean_squared_error(ytest,pred)


        xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
        pkl.dump(reg, open(f'saved_train/reg_{f}_{i}.pkl','wb'))

np.save('saved_train/accu.npy', accu)


