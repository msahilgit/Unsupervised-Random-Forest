import copy as cp
import numpy as np
np.bool = np.bool_
import matplotlib as mt
import matplotlib.pyplot as plt
import pyemma.plots as mplt 
import pyemma.coordinates as coor
import pyemma
import pickle as pkl
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as splt
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
import sklearn.metrics as mtr
from tqdm import tqdm
import sys
sys.path.append('../0_python_modules/')
import extras
import proximity_matrix as pmt
import synthetic_data as syn
import metrics
import gen_data as gd
import navjeet_hist as nh


traj_data = np.load('../1_datasets/p450_bpj/prconf_dists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]
features = np.concatenate((traj_data))[::10]

rmsds = np.load('../1_datasets/p450_bpj/trajs_from_bhuppi/rmsd1_from_bhupi.npz')
rmsds = [ rmsds[i] for i in list(rmsds) ]
rmsds = np.concatenate((rmsds))[::10]

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)


accu = np.zeros(( len(randoms) ))

for i in range(len(randoms)):

    xtrain, xtest, ytrain, ytest = splt( features, rmsds, test_size=0.3, random_state=randoms[i] )

    reg = rfr(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    reg.fit(xtrain, ytrain)

    fimp = reg.feature_importances_
    np.save(f'saved_supervised/fimp_rfr{i}.npy', fimp)
    pkl.dump(reg, open(f'saved_supervised/rfr{i}.pkl', 'wb') )

    pred = reg.predict(xtest)
    accu[i] = mtr.mean_squared_error(ytest, pred)

np.save(f'saved_supervised/accu_rfr.npy', accu)


