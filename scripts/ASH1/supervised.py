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


traj_data = np.load('../1_datasets/ASH1/prconf_dists.npy')[0]
rg = np.loadtxt('../1_datasets/ASH1/rg_ca.xvg', comments=['@','#'])[:,1]
mini = 2.85
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)


accu = np.zeros((len(randoms)))
for i in range(len(randoms)):

    xtrain, xtest, ytrain, ytest = splt(traj_data, rg, test_size=0.3, random_state=randoms[i])
    reg_ = rfr(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    reg_.fit(xtrain, ytrain)
    fimp = reg_.feature_importances_

    pred_ = reg_.predict(xtest)
    accu[i] = mtr.mean_squared_error(ytest, pred_)

    pkl.dump(reg_, open(f'saved_supervised/reg_supervised{i}.pkl','wb'))
    np.save(f'saved_supervised/fimp_supervised{i}.npy', fimp)

    fimp = fimp.argsort()[::-1][:150]
    tic2 = coor.tica( [ traj_data[:,fimp] ], lag=10, dim=2 ).get_output()[0]
    hh = nh.hist_range(tic2, rg, mini=mini, maxi=7)

    np.save(f'saved_supervised/hist_{i}.npy', hh[4])
    np.save(f'saved_supervised/extents_{i}.npy', hh[:4])


np.save('saved_supervised/accu_.npy', accu)




