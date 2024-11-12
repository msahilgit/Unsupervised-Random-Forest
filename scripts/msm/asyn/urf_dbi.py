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

sys.path.append('../../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
import navjeet_hist as nh
import msm_analysis as ana
import idp_analysis as idp


traj_data = np.load('../../../1_datasets/idp_rg/ca_dists.npy')[0]
traj_data = ( traj_data - np.min(traj_data, axis=0) ) / ( np.max(traj_data, axis=0) - np.min(traj_data, axis=0) )
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

n_clus = np.array([20, 30, 50, 70, 100, 150, 250, 350, 500, 700, 1000, 1500])
msm_lags = np.arange(5,70+1,5)
n_pcca = np.array([2, 3, 4, 5, 6])

dbi = np.zeros(( len(randoms), len(n_clus), len(msm_lags), len(n_pcca) )) + np.nan
for i in range(len(randoms)):

    fspace = np.load(f'../../../2_different_system_idp_rg/saved_fidp/fimp_hlabel3_{i}.npy').argsort()[::-1][:200]
    fspace = traj_data[:,fspace]

    for cl in range(len(n_clus)):
        dtrj = np.load(f'saved_msm3/dtraj_{i}_{n_clus[cl]}.npy')

        for ml in range(len(msm_lags)):
            for pc in range(len(n_pcca)):

                try:
                    model = msm.estimate_markov_model( [dtrj], lag=msm_lags[ml])
                    model.pcca(n_pcca[pc])
                    mdis = model.metastable_sets
                    ostates = len(mdis)

                    if np.concatenate((mdis)).shape[0] == n_clus[cl]:
                        print(cl,ml,n_pcca[pc], ostates)
                        mdtrj = ana.get_mdtrj([dtrj], mdis)

                        dbi[i,cl,ml,pc] = mtr.davies_bouldin_score(fspace, mdtrj)


                except:
                    pass
    
np.save('saved_dbi/dbi.npy', dbi)
