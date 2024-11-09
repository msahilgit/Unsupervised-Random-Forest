import copy as cp
import numpy as np
np.bool = np.bool_
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
import extras


features = [ np.load(f'../../../1_datasets/mopR_ensembles/phi_psi_chi1_{i}.npy') for i in ['aw', 'bw', 'am', 'bm'] ]

labels = [i.shape[0] for i in features]
labels = np.concatenate(( [ np.zeros((i))+a for a,i in enumerate(labels) ] ))

features = np.concatenate((features))

randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

accu = np.zeros((len(randoms),2))
for i in range(len(randoms)):

    xtrain, xtest, ytrain, ytest = splt(features, labels, test_size=0.3, random_state=randoms[i])

    clf = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    clf.fit(xtrain, ytrain)

    pred = clf.predict(xtest)

    accu[i,0] = acc(ytest, pred)
    accu[i,1] = extras.get_f1_score( cfm(ytest, pred) )

    pkl.dump(clf, open(f'saved_clf_all/clf_all_{i}.pkl', 'wb'))
    fimp = clf.feature_importances_
    np.save(f'saved_clf_all/fimp_all_{i}.npy', fimp)


