import numpy as np
np.bool = np.bool_
import copy as cp
import sys
import os
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


features = np.load('../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

#create_random = syn.synthetic_data(features, size=1)
#create_random.random()
#pfeatures, plabels = create_random.get_output()

#np.save('saved_synthetic_random/pfeatures.npy', pfeatures)
#np.save('saved_synthetic_random/plabels.npy', plabels)
pfeatures = np.load('saved_synthetic_random/pfeatures.npy')
plabels = np.load('saved_synthetic_random/plabels.npy')

clfs = []
#accu = np.zeros((len(randoms),2))
for i in range(len(randoms)):

    doit = True
    if os.path.exists(f'saved_synthetic_random/clf_random{i}.pkl'):
        clf_ = pkl.load( open(f'saved_synthetic_random/clf_random{i}.pkl','rb') )
        try:
            dd = clf_.apply(features[:10])
            doit=False
        except:
            pass

    if doit:
        xtrain, xtest, ytrain, ytest = splt(pfeatures, plabels, test_size=0.3, random_state=randoms[i])
        clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
        clf_.fit(xtrain, ytrain)
        xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
        pkl.dump(clf_, open(f'saved_synthetic_random/clf_random{i}.pkl','wb'))

    #pred_ = clf_.predict(xtest)
    #accu[i,0] = acc(ytest,pred_)
    #accu[i,1] = extras.get_f1_score( cfm(ytest, pred_) )

    clfs.append(clf_)

#np.save('saved_synthetic_random/accu_random.npy', accu)


random_pmt = pmt.calculate_condenced_pmt_multi( clfs, features, n_jobs=94 )
np.save('saved_synthetic_random/random_pmt_mean.npy', random_pmt)


hc_random = scc.hierarchy.linkage(random_pmt, method='average')
np.save('saved_synthetic_random/hc_random.npy', hc_random)


traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]


#accu = np.zeros((9,len(randoms),2))
for h in range(2,11):
    hlabels_ = extras.get_hc_dtraj(hc_random, nids=h)

    for i in range(len(randoms)):

        doit=True
        if os.path.exists(f'saved_synthetic_random/clf_hlabel{h}_{i}.pkl'):
            try:
                clf_ = pkl.load(open(f'saved_synthetic_random/clf_hlabel{h}_{i}.pkl', 'rb'))
                dd = clf_.apply(features[:10])
                doit=False
            except:
                pass

        if doit:
            xtrain, xtest, ytrain, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
            clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
            clf_.fit(xtrain, ytrain)
            pkl.dump(clf_, open(f'saved_synthetic_random/clf_hlabel{h}_{i}.pkl','wb'))
            xtrain, xtest, ytrain, ytest = 0, 0, 0, 0

        #pred_ = clf_.predict(xtest)
        #accu[h-2,i,0] = acc(ytest,pred_)
        #accu[h-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )

        fimp = clf_.feature_importances_
        np.save(f'saved_synthetic_random/fimp_hlabel{h}_{i}.npy', fimp)

        top200 = fimp.argsort()[::-1][:200]
        tic2 = [traj_data[i][:,top200] for i in range(6)]
        tic2 = coor.tica(tic2, lag=100, dim=2).get_output()
        tic2 = [tic2[0], tic2[4], tic2[3]]
        hist_ = []
        extents = np.zeros((len(dists), 4))
        for k in range(len(dists)):
            hh = nh.hist_range(tic2[k], dists[k], mini=0, maxi=0.6 )
            hist_.append(hh[4])
            extents[k] = hh[:4]
        np.save(f'saved_synthetic_random/hist_{h}_{i}.npy', hist_)
        np.save(f'saved_synthetic_random/extents_{h}_{i}.npy', extents)



#np.save('saved_synthetic_random/accu_hc.npy', accu)


