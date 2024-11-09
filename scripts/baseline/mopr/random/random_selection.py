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

sys.path.append('../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


traj_data = np.load('../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]
dists = np.concatenate(( np.load('../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))

randoms = np.loadtxt('../../1_datasets/randoms.txt', dtype=int)


irandom = []
for i in randoms:
    np.random.seed(i)
    ff = np.random.random(size=229)
    ff = ff/ff.sum()
    irandom.append(ff.argsort()[::-1])
irandom = np.array(irandom)


hists=[]
extents=[]
for i in range(len(randoms)):
    fimp = irandom[i][:40]
    tic2 = [t[:,fimp] for t in traj_data]
    tic2 = np.concatenate(( coor.tica(tic2, lag=10, dim=2).get_output() ))
    hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1)
    hists.append(hh[4])
    extents.append(hh[:4])


np.save('saved_random_selection/hists.npy', hists)
np.save('saved_random_selection/extents.npy', extents)


