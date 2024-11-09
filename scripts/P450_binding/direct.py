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

sys.path.append('../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


traj_data = [ np.load(f'../1_datasets/p450_2018/features{i}.npy') for i in range(3) ]
dists = [ np.loadtxt(f'../1_datasets/p450_2018/distance{i}.xvg', comments=['@','#'])[:,1] for i in range(3) ]

tic2 = coor.tica(traj_data, lag=1000, dim=2).get_output()
hist_ = []
extents = np.zeros((len(dists), 4))
for k in range(len(dists)):
    hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
    hist_.append(hh[4])
    extents[k] = hh[:4]
np.save(f'saved_direct/hist_.npy', hist_)
np.save(f'saved_direct/extents_.npy', extents)


