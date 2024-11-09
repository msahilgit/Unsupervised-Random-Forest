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

sys.path.append('../../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


features = np.load('../../../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)
labels = np.load('../../../1_datasets/t4l/labels.npy')
indices = np.arange(features.shape[0])

nofs = features.shape[1]
accu_train = np.zeros(( len(randoms), nofs, 2 ))
accu_test = np.zeros(( len(randoms), nofs, 2 ))

for i in range(len(randoms)):

    train, test = splt( indices, test_size=0.3, random_state=randoms[i])

    xtrain = features[train]
    ytrain = labels[train]

    clf = rfc(n_estimators=1000, n_jobs=94, random_state=randoms[i])
    clf.fit(xtrain, ytrain)

    for f in range(nofs):

        xtrain = features[train]
        xtrain[:,f] = 10
        ytrain = labels[train]
        pred_train = clf.predict(xtrain)

        accu_train[i,f,0] = acc(ytrain, pred_train)
        accu_train[i,f,1] = extras.get_f1_score( cfm(ytrain, pred_train) )

        xtest = features[test]
        xtest[:,f] = -1
        ytest = labels[test]
        pred_test = clf.predict(xtest)

        accu_test[i,f,0] = acc(ytest, pred_test)
        accu_test[i,f,1] = extras.get_f1_score( cfm(ytest, pred_test) )

        print(i,f)


np.save('saved_predict/accu_train_10.npy', accu_train)
np.save('saved_predict/accu_test_10.npy', accu_test)







