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
import scipy.cluster as scc
import sys
sys.path.append('../0_python_modules/')
import extras
import proximity_matrix as pmt
import synthetic_data as syn
import metrics
import gen_data as gd
import navjeet_hist as nh


traj_data = np.load('../1_datasets/polymer_32beads/alldists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]
features = np.concatenate((traj_data))[::50]

rgs = np.load('../1_datasets/polymer_32beads/rgs/rg.npz')
rgs = [ rgs[i] for i in list(rgs) ]
rgs = np.concatenate((rgs))[::50]
labels = np.zeros((len(rgs)))
labels[np.where(rgs<0.5)[0]] = 1

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)

for i in range(len(randoms)):
    xtrain, xtest, ytrain, ytest = splt(features, labels, test_size=0.3, random_state=randoms[i])
    clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    clf_.fit(xtrain, ytrain)

    pkl.dump(clf_, open(f'saved_supervised/clf_supervised{i}.pkl','wb'))
    fimp = clf_.feature_importances_
    np.save(f'saved_supervised/fimp_supervised{i}.npy', fimp)





