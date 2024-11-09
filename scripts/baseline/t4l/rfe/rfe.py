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
from sklearn.feature_selection import RFE
import sklearn.metrics as mtr
import scipy as sc
import scipy.cluster as scc
from tqdm import tqdm
import time
import pyemma
import pyemma.coordinates as coor
import pyemma.plots as mplt

sys.path.append('../../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


features = np.load('../../../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)
labels = np.load('../../../1_datasets/t4l/labels.npy')
dist0 = np.loadtxt('../../traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])
traj_data = [np.load(f'../../traj_specific_data/features_{i}.npy') for i in range(6)]

hists = []
extents = []
for i in range(len(randoms)):

    _, xtest, _, ytest = splt( features, labels, test_size=0.3, random_state=randoms[i])

    clf = pkl.load( open(f'../../saved_supervised/clf_supervised_{i}.pkl', 'rb') ) 

    selector = RFE(clf, n_features_to_select=200, step=15)
    selector.fit(xtest, ytest)

    fimp = np.where(selector.support_==True)[0]
    np.save(f'saved_rfe/fimp_{i}.npy', fimp)

    tic2 = [i[:,fimp] for i in traj_data]
    tic2 = coor.tica(tic2, lag=700, dim=2).get_output()[0]
    hh = nh.hist_range(tic2, dist0, mini=0, maxi=0.6)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_rfe/hists.npy', hists)
np.save('saved_rfe/extents.npy', extents)
