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

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)

msm_lags = np.arange(5,50+1,5)
methods = ['VAMP1', 'VAMP2', 'VAMPE']

n_jobs = 94

cv=10


vamps_ = np.zeros(( traj_data[0].shape[1], len(msm_lags), len(methods), 2 )) + np.nan

for f in range(traj_data[0].shape[1]):

    for l in range(len(msm_lags)):
    
        np.random.seed(randoms[0])
    
        dd = [[], [], []]
        for c in range(cv):
    
            perm = np.random.permutation( len(traj_data) )
    
            train = [traj_data[i][:,f] for i in perm[:5]]
            test = [traj_data[i][:,f] for i in perm[5:]]
        
            vamp = coor.vamp( train, lag=msm_lags[l] )
        
            for m in range(len(methods)):
                dd[m].append( vamp.score(test, score_method=methods[m]) )
        
        vamps_[f,l,:,0] = np.nanmean(dd, axis=1)
        vamps_[f,l,:,1] = np.nanstd(dd, axis=1)
    

np.save('saved_vamp/vamps_all.npy', vamps_)


