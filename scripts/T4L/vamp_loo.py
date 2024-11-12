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


n_jobs = 94

cv=len(traj_data)


vamps_ = np.zeros(( traj_data[0].shape[1], cv )) + np.nan

for f in range(traj_data[0].shape[1]):

    
    for c in range(cv):
    
        perm = np.delete(np.arange(cv), c)
    
        train = [traj_data[i][:,f] for i in perm]
        test = [traj_data[c][:,f]]
        
        vamp = coor.vamp( train, lag=700 )
        
        vamps_[f,c] = vamp.score(test, score_method='VAMP2') 
    print(f)
        

np.save('saved_vamp/vamps_all_loo.npy', vamps_)



