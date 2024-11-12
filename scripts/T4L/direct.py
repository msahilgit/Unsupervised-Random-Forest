import numpy as np
np.bool = np.bool_
import copy as cp
import sys
import matplotlib as mt
import matplotlib.pyplot as plt
import pickle as pkl
#import sklearn as skl
#from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr
#from sklearn.model_selection import train_test_split as splt
#from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
#import sklearn.metrics as mtr
#import scipy as sc
#import scipy.cluster as scc
from tqdm import tqdm
import time
import pyemma
import pyemma.coordinates as coor
import pyemma.plots as mplt

sys.path.append('../0_python_modules/')

#import proximity_matrix as pmt
#import synthetic_data as syn
#import extras
#from navjeet_hist import hist_raw_data
import navjeet_hist as nh


#print(1)
#tic2 = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
#dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]
#print(2)
#
#
#tic2 = coor.tica(tic2, lag=100, dim=2).get_output()
#tic2 = [tic2[0], tic2[4], tic2[3]]
#print(3)
#
#hist_ = []
#extents = np.zeros((len(dists), 4))
#for k in range(len(dists)):
#    hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
#    hist_.append(hh[4])
#    extents[k] = hh[:4]
#
#np.save(f'saved_direct/hist.npy', hist_)
#np.save(f'saved_direct/extents.npy', extents)


#############################
#############################

#tic2 = np.load('traj_specific_data/tica_jctc.npy', allow_pickle=True)

#hist_ = []
#extents = np.zeros((len(dists), 4))
#for k in range(len(dists)):
#    hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
#    hist_.append(hh[4])
#    extents[k] = hh[:4]

#np.save(f'saved_direct/hist.npy', hist_)
#np.save(f'saved_direct/extents.npy', extents)



#############################
#############################

traj_data = [np.load(f'saved_direct/traj_data/0000000{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]

tlags = np.array([ 10, 20, 30, 50, 70, 100, 150, 200, 250, 350, 500, 700, 1000, 1500, 2000 ])

for  t in tlags:
    tic2 = coor.tica( traj_data, lag=t, dim=2 ).get_output()
    tic2 = [ tic2[0], tic2[4], tic2[3] ]

    hist_ = []
    extents = []
    for k in range(len(dists)):
        hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
        hist_.append(hh[4])
        extents.append(hh[:4])

    np.save(f'saved_direct/tlag_{t}_hist.npy', hist_)
    np.save(f'saved_direct/tlag_{t}_extents.npy', extents)

    print(t)


