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


features = np.load('../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

#create_marginal = syn.synthetic_data(features, size=1)
#create_marginal.marginal()
#pfeatures, plabels = create_marginal.get_output()

#np.save('saved_synthetic_marginal/pfeatures.npy', pfeatures)
#np.save('saved_synthetic_marginal/plabels.npy', plabels)

clfs = []
#accu = np.zeros((len(randoms),2))
for i in range(len(randoms)):
    #xtrain, xtest, ytrain, ytest = splt(pfeatures, plabels, test_size=0.3, random_state=randoms[i])
    #clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    #clf_.fit(xtrain, ytrain)

    #pred_ = clf_.predict(xtest)
    #accu[i,0] = acc(ytest,pred_)
    #accu[i,1] = extras.get_f1_score( cfm(ytest, pred_) )


    #xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
    #clfs.append(clf_)
    #pkl.dump(clf_, open(f'saved_synthetic_marginal/clf_marginal{i}.pkl','wb'))
    clfs.append( pkl.load(open(f'saved_synthetic_marginal/clf_marginal{i}.pkl','rb')) )

#np.save('saved_synthetic_marginal/accu_marginal.npy', accu)


marginal_pmt = pmt.calculate_condenced_pmt_multi( clfs, features, n_jobs=32 )
np.save('saved_synthetic_marginal/marginal_pmt_mean.npy', marginal_pmt)


hc_marginal = scc.hierarchy.linkage(marginal_pmt, method='average')
np.save('saved_synthetic_marginal/hc_marginal.npy', hc_marginal)


traj_data = [np.load(f'traj_specific_data/features_{i}.npy') for i in range(6)]
dists = [np.loadtxt(f'traj_specific_data/distances{i}.xvg', comments=['@','#'], usecols=[3]) for i in [0,4,3]]


accu = np.zeros((9,len(randoms),2))
for h in range(2,11):
    hlabels_ = extras.get_hc_dtraj(hc_marginal, nids=h)

    for i in range(len(randoms)):
        xtrain, xtest, ytrain, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
        clf_.fit(xtrain, ytrain)

        pred_ = clf_.predict(xtest)
        accu[h-2,i,0] = acc(ytest,pred_)
        accu[h-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )

        xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
        pkl.dump(clf_, open(f'saved_synthetic_marginal/clf_hlabel{h}_{i}.pkl','wb'))
        fimp = clf_.feature_importances_
        np.save(f'saved_synthetic_marginal/fimp_hlabel{h}_{i}.npy', fimp)

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
        np.save(f'saved_synthetic_marginal/hist_{h}_{i}.npy', hist_)
        np.save(f'saved_synthetic_marginal/extents_{h}_{i}.npy', extents)



np.save('saved_synthetic_marginal/accu_hc.npy', accu)


