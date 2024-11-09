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
randoms = np.loadtxt('../../../../1_datasets/randoms.txt', dtype=int)


asupervised = np.load('../../saved_supervised/accu_supervised.npy')


features = np.concatenate((traj_data))[::20]
nofs = features.shape[1]
labels = dists[::20]
accu = np.zeros(( nofs, len(randoms) ))
for f in range(nofs):
    sfeatures = np.delete( features, f, axis=1 )

    for i in range(len(randoms)):
        _, xtest, _, ytest = splt(sfeatures, labels, test_size=0.3, random_state=randoms[i])
        reg = pkl.load(open(f'saved_train/reg_{f}_{i}.pkl', 'rb'))
        pred = reg.predict(xtest)
        accu[f,i] = mtr.mean_squared_error(ytest,pred)
        xtrain, xtest, ytrain, ytest = 0, 0, 0, 0

np.save('saved_train/accu.npy', accu)


accu = asupervised - accu
hists = []
extents = []
for i in range(len(randoms)):
    fimp = accu[:,i].argsort()[:40]               # here no [::-1] as larger mse is bad
    tic2 = [i[:,fimp] for i in traj_data]
    tic2 = np.concatenate(( coor.tica(tic2, lag=10, dim=2).get_output() ))
    hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_train/hists.npy', hists)
np.save('saved_train/extents.npy', extents)


