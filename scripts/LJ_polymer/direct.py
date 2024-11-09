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


traj_data = np.load('../1_datasets/polymer_32beads/alldists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]

rgs = np.load('../1_datasets/polymer_32beads/rgs/rg.npz')
rgs = [ rgs[i] for i in list(rgs) ]
rgs = np.concatenate((rgs))

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)


tic2 = np.concatenate(( coor.tica(traj_data, lag=10, dim=2).get_output() ))

hh = nh.hist_range(tic2, rgs, mini=0, maxi=0.5)

np.save('saved_direct/hist.npy', hh[4])
np.save('saved_direct/extents.npy', hh[:4])


