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


features = np.load('../1_datasets/idp_rg/ca_dists.npy')[0]
rg_ca = np.loadtxt('../1_datasets/idp_rg/rg_ca.xvg', comments=['@','#'])[:,1]
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

hists=[]
extents=[]
for h in range(2,11):

    h1=[]
    e1=[]
    for i in range(len(randoms)):
        fimp = np.load(f'../2_different_system_idp_rg/saved_fidp/fimp_hlabel{h}_{i}.npy')

        top200 = fimp.argsort()[::-1][:200]
        tic2 = coor.tica([features[:,top200]], lag=10, dim=2).get_output()
        hh = nh.hist_range(tic2[0], rg_ca, mini=3, maxi=8 )
        h1.append(hh[4])
        e1.append(hh[:4])
    hists.append(h1)
    extents.append(e1)

np.save('saved_urf/hists.npy', hists)
np.save('saved_urf/extents.npy', extents)

