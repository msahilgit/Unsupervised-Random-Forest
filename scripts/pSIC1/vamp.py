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

import warnings
warnings.filterwarnings("ignore", message="Call to deprecated function")


traj_data = np.load('../1_datasets/PSIC1/prconf_dists.npy')[0]
nobs = traj_data.shape[0]

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)

msm_lags = np.arange(5,70+1,5)
mini=0.2
maxi=0.3

n_jobs = 94

cv=10


vamps_ = np.zeros(( traj_data.shape[1], len(msm_lags), 2 )) + np.nan

for f in range(traj_data.shape[1]):

    for l in range(len(msm_lags)):
    
        np.random.seed(randoms[0])
    
        dd = []
        for c in range(cv):
    
            perm = np.random.random() * (maxi-mini) + mini
            perm = int(nobs * (1-perm))
    
            train = [ traj_data[:,f][:perm] ]
            test = [ traj_data[:,f][perm:] ]
        
            vamp = coor.vamp( train, lag=msm_lags[l] )
        
            dd.append( vamp.score(test, score_method='VAMP2') )

        print(f,l)
        
        vamps_[f,l,0] = np.nanmean(dd)
        vamps_[f,l,1] = np.nanstd(dd)
    

np.save('saved_vamp/vamps_all.npy', vamps_)


