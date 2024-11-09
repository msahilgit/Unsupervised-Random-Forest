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


nofs = features.shape[1]
accu_train = np.zeros(( nofs, len(randoms), 2 ))
accu_test = np.zeros(( nofs, len(randoms), 2 ))

for f in range(nofs):

    sfeatures = np.delete( features, f, axis=1 )

    for i in range(len(randoms)):

        xtrain, xtest, ytrain, ytest = splt(sfeatures, labels, test_size=0.3, random_state=randoms[i])

        clf = rfc(n_estimators=1000, n_jobs=94, random_state=randoms[i])
        clf.fit(xtrain, ytrain)

        pred = clf.predict(xtest)
        accu_test[f,i,0] = acc(ytest, pred)
        accu_test[f,i,1] = extras.get_f1_score( cfm(ytest, pred) )

        pred = clf.predict(xtrain)
        accu_train[f,i,0] = acc(ytrain, pred)
        accu_train[f,i,1] = extras.get_f1_score( cfm(ytrain, pred) )

        print(f,i)

np.save('saved_training/accu_train.npy', accu_train)
np.save('saved_training/accu_test.npy', accu_test)







