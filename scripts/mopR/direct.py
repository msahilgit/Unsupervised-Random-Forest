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

sys.path.append('../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh


traj_data = np.load('../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data)]
dists = np.concatenate((  np.load('../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))
#
#
#tic2 = np.concatenate(( coor.tica( traj_data, lag=10, dim=2 ).get_output() ))
#
#hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1 )
#
#np.save(f'saved_direct/hist.npy', hh[4])
#np.save(f'saved_direct/extents.npy', hh[:4])

################################################################
# for bound state
################################################################


#tic2 = np.concatenate(( coor.tica( traj_data, lag=10, dim=2 ).get_output() ))

#hh = nh.hist_range(tic2, dists, mini=0, maxi=0.35 )
#
#np.save(f'saved_direct/different_state_bound/hist.npy', hh[4])
#np.save(f'saved_direct/different_state_bound/extents.npy', hh[:4])




################################################################
# for unbound state
################################################################


tic2 = np.concatenate(( coor.tica( traj_data, lag=10, dim=2 ).get_output() ))

hh = nh.hist_range(tic2, dists, mini=1, maxi=10 )

np.save(f'saved_direct/different_state_unbound/hist.npy', hh[4])
np.save(f'saved_direct/different_state_unbound/extents.npy', hh[:4])
