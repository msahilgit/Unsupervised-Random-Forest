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
import scipy.cluster as scc
import sys
sys.path.append('../0_python_modules/')
import extras
import proximity_matrix as pmt
import synthetic_data as syn
import metrics
import gen_data as gd
import navjeet_hist as nh
import metrics as mtr


traj_data = np.load('../1_datasets/p450_bpj/prconf_dists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]
features = np.concatenate((traj_data))[::5]

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)


hc_permute = np.load('saved_furf/hc_permute.npy')


lc = np.zeros((9,len(randoms))) + np.nan
for h in range(2,11):
    hlabels_ = extras.get_hc_dtraj(hc_permute, nids=h)

    for i in range(len(randoms)):
        _, xtest, _, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])

        clf = pkl.load( open(f'saved_furf/clf_hlabel{h}_{i}.pkl','rb'))

        lc[h-2,i] = mtr.learning_coefficient(hlabels_, clf, xtest, ytest)
        print(h,i)


np.save('saved_furf/lc1.npy', lc)






