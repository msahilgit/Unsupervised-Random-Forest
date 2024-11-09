import numpy as np
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
import shap


features = np.load('s229_dists.npz')
features = np.concatenate(( [features[i] for i in list(features)] ))[::20]
randoms = np.loadtxt('randoms.txt', dtype=int)
labels = np.concatenate(( np.load('distance_alldata.npy', allow_pickle=True) ))[::20]


for i in range(4, -1, -1):

    _, xtest, _, _ = splt(features, labels, test_size=0.3, random_state=randoms[i])

    clf = pkl.load( open(f'reg_supervised{i}.pkl', 'rb') )

    ss = shap.Explainer(clf.predict, xtest)
    sv = ss.shap_values(xtest)

    np.save(f'saved_shap/shap_values_supervised{i}.npy', sv)
