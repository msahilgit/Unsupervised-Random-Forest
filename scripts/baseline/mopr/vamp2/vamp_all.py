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

sys.path.append('../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
import navjeet_hist as nh
import msm_analysis as ana

import warnings
warnings.filterwarnings("ignore", message="Call to deprecated function")


traj_data = np.load('../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data) ]


randoms = np.loadtxt('../../1_datasets/randoms.txt').astype(int)

n_jobs = 94

cv=10


vamps_ = np.zeros(( traj_data[0].shape[1], cv )) + np.nan

for f in range(traj_data[0].shape[1]):

    
    np.random.seed(randoms[0])
    
    for c in range(cv):
    
        perm = np.random.permutation( len(traj_data) )
    
        train = [traj_data[i][:,f] for i in perm[:100]]
        test = [traj_data[i][:,f] for i in perm[100:]]
        
        vamp = coor.vamp( train, lag=10 )
        
        vamps_[f,c] = vamp.score(test, score_method='VAMP2') 

    print(f)
        
    

np.save('saved_vamp/vamps_all.npy', vamps_)



