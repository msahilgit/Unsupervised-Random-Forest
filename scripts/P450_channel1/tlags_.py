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

labels = np.load('../1_datasets/p450_bpj/trajs_from_bhuppi/rmsd1_from_bhupi.npz')
labels = [ labels[i] for i in list(labels) ]
labels = np.concatenate((labels))

tlags=[10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500, 700, 1000, 1500, 2000]
randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)



hists = []
extents = []
for t in tlags:

    tic2 = np.concatenate(( coor.tica(traj_data, lag=t, dim=2).get_output() ))

    hh = nh.hist_range( tic2, labels, mini=0, maxi=0.2)

    hists.append(hh[4])
    extents.append(hh[:4])


np.save('saved_direct/tlags.npy', hists)
np.save('saved_direct/extents_tlags.npy', extents)


