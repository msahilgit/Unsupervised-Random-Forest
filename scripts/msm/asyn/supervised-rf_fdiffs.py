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

randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

n_clus = np.array([20, 30, 50, 70, 100, 150, 250, 350, 500, 700, 1000, 1500])
msm_lags = np.arange(5,70+1,5)
n_pcca = np.array([2, 3, 4, 5, 6])

n_jobs = 94

dtrajs = dtrajs = [ [] for i in range(len(randoms)) ]
to_do_clus = np.ones(( len(randoms), len(n_clus) )).astype(bool)
tic2s = []
to_do_random = np.ones(randoms.shape).astype(bool)

cmean_strict = np.zeros(( len(randoms), len(n_clus), len(msm_lags), len(n_pcca) )) + np.nan
cmean_loose = np.zeros(( len(randoms), len(n_clus), len(msm_lags), len(n_pcca) )) + np.nan
cmedian_strict = np.zeros(( len(randoms), len(n_clus), len(msm_lags), len(n_pcca) )) + np.nan
cmedian_loose = np.zeros(( len(randoms), len(n_clus), len(msm_lags), len(n_pcca) )) + np.nan
for p in range(len(n_pcca)):


    for r in range(len(randoms)):
        fimp = np.load(f'../../../2_different_system_idp_rg/saved_supervised/fimp_supervised_{r}.npy')
        fimp = fimp.argsort()[::-1][:200]

        for n in range(len(n_clus)):
    
            if to_do_clus[r,n]:
                dtrj = np.load(f'saved_msm/dtraj_{r}_{n_clus[n]}.npy')
                dtrajs[r].append(dtrj)
                to_do_clus[r,n] = False
    
            dtrj = dtrajs[r][n]
    
    
            for m in range(len(msm_lags)):
    
                try:
                    model = msm.estimate_markov_model( [dtrj], lag=msm_lags[m] )
         
                    model.pcca(n_pcca[p])
                    mdis = model.metastable_sets
                    ostates = len(mdis)
    
                    if np.concatenate((mdis)).shape[0] == n_clus[n]:
                        print(r,n,m,n_pcca[p], ostates)
            
                        mdtrj = cp.deepcopy(dtrj)
                        for i in range(len(mdis)):
                            for j in mdis[i]:
                                mdtrj[ np.where( dtrj == j )[0] ] = i
            
    
                        cmean_loose[r,n,m,p], cmean_strict[r,n,m,p], cmedian_loose[r,n,m,p], cmedian_strict[r,n,m,p] = idp.get_contact_diffs(mdtrj, traj_data[:,fimp], cutoff=0.5)
    
            
                except:
                    pass



np.save(f'saved_fdiffs/cmean_loose.npy', cmean_loose)
np.save(f'saved_fdiffs/cmean_strict.npy', cmean_strict)
np.save(f'saved_fdiffs/cmedian_loose.npy', cmedian_loose)
np.save(f'saved_fdiffs/cmedian_strict.npy', cmedian_strict)


