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



paths = ['furf', 'synthetic_marginal', 'synthetic_random', 'synthetic_nonsense']
names1 = ['average', 'marginal', 'random', 'nonsense']
names2 = ['permute', 'marginal', 'random', 'nonsense']

corr = np.zeros((len(paths))) + np.nan
for p in range(len(paths)):

    hc = np.load(f'saved_{paths[p]}/hc_{names1[p]}.npy')

    dcop = scc.hierarchy.cophenet(hc)
    dpmt = np.load(f'saved_{paths[p]}/{names2[p]}_pmt_mean.npy')

    mcop = np.mean(dcop)
    mpmt = np.mean(dpmt)

    f1 = np.sum( (dcop-mcop)*(dpmt-mpmt) )
    f2 = np.sum((dcop-mcop)**2)
    f3 = np.sum( (dpmt-mpmt)**2 )

    corr[p] = f1 / np.sqrt(f2*f3)


np.save('saved_cophenet/corr.npy', corr)
