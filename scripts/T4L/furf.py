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

#features = np.load('../1_datasets/t4l/features.npy')
#randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)
#
#
#accu = np.zeros((9,5,2))
#
#for l in range(2,11):
#
#    hlabels_ = np.load(f'saved_furf/hlabels_{l}.npy')
#
#    for i in range(len(randoms)):
#
#        xtrain, xtest, ytrain, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
#        clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
#        clf_.fit(xtrain, ytrain)
#
#        pred_ = clf_.predict(xtest)
#        accu[l-2,i,0] = acc(ytest, pred_)
#        try:
#            accu[l-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )
#        except:
#            accu[l-2,i,1] = 0
#
#        pkl.dump(clf_, open(f'saved_furf/clf_hlabel{l}_{i}.pkl', 'wb'))
#        fimp = clf_.feature_importances_
#        np.save(f'saved_furf/fimp_hlabel{l}_{i}.npy', fimp)
#
#np.save('saved_furf/accu_hc.npy', accu)





##############################################################
#   TLAGS
##############################################################



randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

tlags = np.array([ 10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500, 700, 1000, 1500, 2000 ])

traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]


for l in range(2,11):

    for i in range(len(randoms)):

        fimp = np.load(f'saved_furf/fimp_hlabel{l}_{i}.npy')
        top200 = fimp.argsort()[::-1][:200]

        for t in tlags:

            tic2 = [traj_data[i][:,top200] for i in range(6)]
            tic2 = coor.tica(tic2, lag=t, dim=2).get_output()
            tic2 = [tic2[0], tic2[4], tic2[3]]

            hist_ = []
            extents = np.zeros((len(dists), 4))
            for k in range(len(dists)):
                hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
                hist_.append(hh[4])
                extents[k] = hh[:4]

            np.save(f'saved_furf/tlags/hist_{l}_{i}_{t}.npy', hist_)
            np.save(f'saved_furf/tlags/extents_{l}_{i}_{t}.npy', extents)

