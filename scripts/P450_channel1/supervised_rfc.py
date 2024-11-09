import copy as cp
import numpy as np
np.bool = np.bool_
import matplotlib as mt
import matplotlib.pyplot as plt
import pyemma.plots as mplt 
import pyemma.coordinates as coor
import pyemma
import pickle as pkl
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as splt
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
import sklearn.metrics as mtr
from tqdm import tqdm
import sys
sys.path.append('../0_python_modules/')
import extras
import proximity_matrix as pmt
import synthetic_data as syn
import metrics
import gen_data as gd
import navjeet_hist as nh


traj_data = np.load('../1_datasets/p450_bpj/prconf_dists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]
features = np.concatenate((traj_data))[::5]

labels = np.load('../1_datasets/p450_bpj/labels.npz')
labels = [labels[i] for i in list(labels)]
labels = np.concatenate((labels))[::5]

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)


accu = np.zeros(( len(randoms),2 ))

for i in range(len(randoms)):

    xtrain, xtest, ytrain, ytest = splt( features, labels, test_size=0.3, random_state=randoms[i] )

    clf = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    clf.fit(xtrain, ytrain)

    fimp = clf.feature_importances_
    np.save(f'saved_supervised/fimp_rfc{i}.npy', fimp)
    pkl.dump(clf, open(f'saved_supervised/rfc{i}.pkl', 'wb') )

    pred = clf.predict(xtest)
    accu[i,0] = acc(ytest, pred)
    accu[i,1] = extras.get_f1_score( cfm(ytest, pred) )

np.save(f'saved_supervised/accu_rfc.npy', accu)


