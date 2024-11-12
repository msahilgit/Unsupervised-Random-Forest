import numpy as np
np.bool = np.bool_
import copy as cp
import sys
import matplotlib as mt
import matplotlib.pyplot as plt
import pickle as pkl
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split as splt
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
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



randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)
traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]

iters=20

for l in range(2,11):

    for i in range(len(randoms)):

        fimp = np.load(f'saved_furf/fimp_hlabel{l}_{i}.npy')
        fimp = fimp.argsort()

        np.random.seed(randoms[i])
        for n in range(iters):

            tic2 = np.random.choice(fimp, 200)
            tic2 = [traj_data[i][:,tic2] for i in range(6)]
            tic2 = coor.tica(tic2, lag=700, dim=2).get_output()
            tic2 = [tic2[0], tic2[4], tic2[3]]

            hist_ = []
            for k in range(len(dists)):
                hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
                hist_.append(hh[4])

            np.save(f'saved_fimp/random/tlag700/hist_{l}_{i}_{n}.npy', hist_)
            print(l,i,n)

