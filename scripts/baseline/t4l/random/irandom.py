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


randoms = np.loadtxt('../../1_datasets/randoms.txt', dtype=int)

traj_data = [np.load(f'../traj_specific_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt(f'../traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])

fimp = np.load('saved_fimp/irandom.npy')

hists = []
extents = []
for i in range(len(randoms)):

        top200 = fimp[i][:200]
        tic2 = [traj_data[i][:,top200] for i in range(6)]
        tic2 = coor.tica(tic2, lag=700, dim=2).get_output()[0]

        hh = nh.hist_range(tic2, dists, mini=0, maxi=0.6 )
        hists.append(hh[4])
        extents.append( hh[:4] )


np.save('saved_fimp/hists_irandom.npy', hists)
np.save('saved_fimp/extents_irandom.npy', extents)

