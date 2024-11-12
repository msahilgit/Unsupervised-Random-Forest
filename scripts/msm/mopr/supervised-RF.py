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


traj_data = np.load('../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data) ]


labels = np.load('../../1_datasets/mopR_bindings/labels.npy').astype(int)
weights = np.array([ np.where(labels==i)[0].shape[0]/labels.shape[0] for i in np.unique(labels) ])

randoms = np.loadtxt('../../1_datasets/randoms.txt', dtype=int)

n_clus = np.array([20, 30, 50, 70, 100, 150, 250, 350, 500, 700, 1000, 1500])
msm_lags = np.arange(5,50+1,5)
n_pcca = np.array([3, 4, 5, 6])

n_jobs = 94

dtrajs = [ [] for i in randoms]
to_do_clus = np.ones(( len(randoms), len(n_clus) )).astype(bool)
tic2s = []
to_do_random = np.ones(randoms.shape).astype(bool)


for p in range(len(n_pcca)):

    ginis = np.zeros(( len(randoms), len(n_clus), len(msm_lags), n_pcca[p] )) + np.nan
    pops = np.zeros(( len(randoms), len(n_clus), len(msm_lags), n_pcca[p] )) + np.nan
    ids = np.zeros(( len(randoms), len(n_clus), len(msm_lags), n_pcca[p], 2 )) + np.nan
    kinetics = np.zeros(( len(randoms), len(n_clus), len(msm_lags), n_pcca[p], n_pcca[p] )) + np.nan

    for r in range(len(randoms)):
        if to_do_random[r]:
            fimp = np.load(f'../../2_additional_s229_mopR/cog/saved_supervised/fimp_supervised_{r}.npy')
            fimp = fimp.argsort()[::-1][:40]
            tic2 = [ i[:,fimp] for i in traj_data ]
            tic2 = coor.tica( tic2, lag=10, dim=2 ).get_output()
            tic2s.append(tic2)
            to_do_random[r] = False

        tic2 = tic2s[r]

        for n in range(len(n_clus)):
    
            if to_do_clus[r,n]:
                clus = km(n_clusters=n_clus[n], init='k-means++', n_init=1, max_iter=5000, random_state=0)
                clus.fit( np.concatenate((tic2)) )
                cdtrj = clus.labels_
                dtrj, k = [], 0
                for i in range(len(tic2)):
                    dd = []
                    for j in range(len(tic2[i])):
                        dd.append(cdtrj[k])
                        k = k+1
                    dtrj.append(dd)
                dtrajs[r].append( dtrj )
                np.savez(f'saved_msm/dtraj_{r}_{n_clus[n]}.npz', *dtrj)
                to_do_clus[r,n] = False
    
            dtrj = dtrajs[r][n]
            cdtrj = np.concatenate(( dtrj ))
    
    
            for m in range(len(msm_lags)):
    
                try:
                    model = msm.estimate_markov_model( dtrj, lag=msm_lags[m] )
    
                    model.pcca(n_pcca[p])
                    mdis = model.metastable_sets
                    mdis = [i for i in mdis if len(i) > 0]
                    ostates = len(mdis)
    
                    if np.concatenate((mdis)).shape[0] == n_clus[n]:
                        print(r,n,m,n_pcca[p],ostates)
    
                        mdtrj = cp.deepcopy(cdtrj)
                        for i in range(len(mdis)):
                            pops[r,n,m,i] = model.pi[mdis[i]].sum()
                            for j in mdis[i]:
                                mdtrj[ np.where( cdtrj == j )[0] ] = i
    
                        pis, ids[r,n,m][:ostates] = ana.get_pis(mdtrj, labels, ids=True)
                        pis = ana.get_reweighted_pis(pis, weights)
                        ginis[r,n,m][:ostates] = ana.get_gini(pis)
                        entropys[r,n,m][:ostates] = ana.get_entropy(pis)
    
                        for i in range(len(mdis)):
                           for j in range(len(mdis)):
                               if i != j:
                                   kinetics[r,n,m,i,j] = model.mfpt(mdis[i], mdis[j])*msm_lags[m]
    
                except:
                    pass
    
    np.save(f'saved_msm/pops_{n_pcca[p]}.npy', pops)
    np.save(f'saved_msm/ids_{n_pcca[p]}.npy', ids)
    np.save(f'saved_msm/kinetics_{n_pcca[p]}.npy', kinetics)
    np.save(f'saved_msm/ginis_{n_pcca[p]}.npy', ginis)

