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


traj_data = np.load('../1_datasets/ASH1/prconf_dists.npy')[0]
rg=np.loadtxt('../1_datasets/ASH1/rg_ca.xvg', comments=['@','#'])[:,1]
mini=2.85

tlags=[10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500, 700, 1000, 1500, 2000]

hists=[]
for t in tlags:
    tic2 = coor.tica( [traj_data], lag=t, dim=2 ).get_output()[0]

    hists.append(nh.hist_range(tic2, rg, mini=mini, maxi=7)[4])

np.save('saved_direct/tlags.npy', hists)




