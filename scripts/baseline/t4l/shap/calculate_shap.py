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
import shap


features = np.load('../../../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)
labels = np.load('../../../1_datasets/t4l/labels.npy')


for i in range(len(randoms)):

    xtrain, xtest, ytrain, ytest = splt(features, labels, test_size=0.3, random_state=randoms[i])

    clf = pkl.load( open(f'../../../2_jctc_urf/saved_supervised/clf_supervised_{i}.pkl', 'rb') )

    ss = shap.Explainer(clf.predict, xtest)
    sv = ss.shap_values(xtest)

    np.save(f'saved_shap/shap_values_supervised{i}.npy', sv)









