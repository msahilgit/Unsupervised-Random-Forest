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



randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)
traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]

hlabels = np.load('saved_furf/hc_average.npy')
hlabels = extras.get_hc_dtraj(hlabels, nids=2)
hlabels = hlabels[:traj_data[0][::10].shape[0]]
states = np.array([ [i,np.where(hlabels==i)[0].shape[0]] for i in np.unique(hlabels)])
states = states[np.argmin(states[:,1])][0]
gap = 0.5


nofs = traj_data[0].shape[1]
numbers = np.arange(0.02, 1.01, 0.02)

hists = []
extents = []
for i in range(len(randoms)):

    fimp = np.load(f'saved_furf/fimp_hlabel2_{i}.npy')
    fimp = fimp.argsort()[::-1]

    h1 = []
    e1 = []
    for n in numbers:

        tofs = int( nofs * n )
        tofs = fimp[:tofs]

        tic2 = [traj_data[i][:,tofs] for i in range(6)]
        tic2 = coor.tica(tic2, lag=700, dim=2).get_output()
        tic2 = tic2[0][::10]

        hh = nh.hist_range( tic2, hlabels, mini=states-gap, maxi=states+gap)

        print(i,n)
        h1.append(hh[4])
        e1.append(hh[:4])
    hists.append(h1)
    extents.append(e1)

np.save('saved_fimp_hlabel1_numbers/hists.npy', hists)
np.save('saved_fimp_hlabel1_numbers/extents.npy', extents)

