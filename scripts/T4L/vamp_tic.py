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
from sklearn.cluster import KMeans as km
import scipy as sc
import scipy.cluster as scc
from tqdm import tqdm
import time
import pyemma
import pyemma.msm as msm
import pyemma.coordinates as coor
import pyemma.plots as mplt

sys.path.append('../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
import navjeet_hist as nh
import msm_analysis as ana


traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)

vamps = np.load('saved_vamp/vamps_all_loo.npy')

cv=len(traj_data)

n_jobs = 94

hists = []
extents = []
for i in range(cv):

    fimp = vamps[:,i].argsort()[::-1][:200]

    tic2 = [ t[:,fimp] for t in traj_data ]
    tic2 = coor.tica( tic2, lag=700, dim=2).get_output()
    tic2 = [ tic2[0], tic2[4], tic2[3] ]

    h1 = []
    e1 = []
    for k in range(len(dists)):
        hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
        h1.append(hh[4])
        e1.append(hh[:4])

    hists.append(h1)
    extents.append(e1)

np.save('saved_vamp/hists.npy', hists)
np.save('saved_vamp/extents.npy', extents)

