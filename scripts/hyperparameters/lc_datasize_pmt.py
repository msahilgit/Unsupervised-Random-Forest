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
import metrics as mtr


features = np.load('../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

data_sizes = np.arange( 0.1, 1.1, 0.1 )
nobs = features.shape[0]
np.random.seed(randoms[0])
rorder = np.random.permutation(nobs)

lc1 = np.zeros((10,9,5)) + np.nan   # a=1,   b=0
lc2 = np.zeros((10,9,5)) + np.nan   # a=0.8, b=0.2

for s in data_sizes:

    tsize = int( nobs * s )
    tfeatures = features[ rorder[:tsize] ]
    iters=int(s*10)
    hc_ds = np.load(f'saved_data_size_pmt/hc_permute_{iters}.npy')

    for h in range(2,11):
        hlabels_ = extras.get_hc_dtraj(hc_ds, nids=h)

        for i in range(len(randoms)):
            _, xtest, _, ytest = splt(tfeatures, hlabels_, test_size=0.3, random_state=randoms[i])

            clf = pkl.load( open(f'saved_data_size_pmt/clf_s{iters}_hlabel{h}_{i}.pkl','rb') )

            lc1[iters-1,h-2,i] = mtr.learning_coefficient(hlabels_, clf, xtest, ytest, 
                    cfactor=2, b=0, a=1, penalty=0)
            lc2[iters-1,h-2,i] = mtr.learning_coefficient(hlabels_, clf, xtest, ytest, 
                    cfactor=2, b=0.2, a=0.8, penalty=0)


            print(iters,h,i)
np.save('saved_lc/lc1_data_size_pmt.npy', lc1)
np.save('saved_lc/lc2_data_size_pmt.npy', lc2)


