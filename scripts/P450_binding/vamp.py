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


traj_data = [ np.load(f'../1_datasets/p450_2018/features{i}.npy') for i in range(3) ]

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)

msm_lags = np.arange(5,50+1,5)

n_jobs = 94


vamps_ = np.zeros(( traj_data[0].shape[1], len(msm_lags) )) + np.nan

for f in range(traj_data[0].shape[1]):

    for l in range(len(msm_lags)):
    
        dd = [i[:,f] for i in traj_data]

        vamp = coor.vamp( dd, lag=msm_lags[l] )
        
        vamps_[f,l] = vamp.score(dd, score_method='VAMP2')
        
        print(f,l)
np.save('saved_vamp/vamps_all.npy', vamps_)



