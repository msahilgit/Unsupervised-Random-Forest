import copy as cp
import numpy as np
np.bool = np.bool_
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
sys.path.append('../../../0_python_modules/')
import proximity_matrix as pmt
import synthetic_data as syn
import extras
import navjeet_hist as nh
import msm_analysis as ana



features = [ np.load(f'../../../1_datasets/mopR_ensembles/coordinates_{i}.npy') for i in ['aw', 'bw', 'am', 'bm'] ]
features = np.concatenate((features))

randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

np.random.seed(randoms[4])
create_permute = syn.synthetic_data(features, size=1)
create_permute.permute()
pfeatures, plabels = create_permute.get_output()

clfs = []
accu = np.zeros((len(randoms),2))
for i in range(len(randoms)):
    xtrain, xtest, ytrain, ytest = splt(pfeatures, plabels, test_size=0.3, random_state=randoms[i])
    clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    clf_.fit(xtrain, ytrain)

    pred_ = clf_.predict(xtest)
    accu[i,0] = acc(ytest,pred_)
    accu[i,1] = extras.get_f1_score( cfm(ytest, pred_) )


    xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
    clfs.append(clf_)
    pkl.dump(clf_, open(f'saved_urf_coords/clf_permute{i}.pkl','wb'))

np.save('saved_urf_coords/accu_permute.npy', accu)


permute_pmt = pmt.calculate_condenced_pmt_multi( clfs, features, n_jobs=94 )
np.save('saved_urf_coords/permute_pmt_mean.npy', permute_pmt)

hc_permute = scc.hierarchy.linkage(permute_pmt, method='average')
np.save('saved_urf_coords/hc_permute.npy', hc_permute)


accu = np.zeros((9,len(randoms),2))
for h in range(2,11):
    hlabels_ = extras.get_hc_dtraj(hc_permute, nids=h)

    for i in range(len(randoms)):
        xtrain, xtest, ytrain, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
        clf_.fit(xtrain, ytrain)

        pred_ = clf_.predict(xtest)
        accu[h-2,i,0] = acc(ytest,pred_)
        accu[h-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )

        xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
        pkl.dump(clf_, open(f'saved_urf_coords/clf_hlabel{h}_{i}.pkl','wb'))
        fimp = clf_.feature_importances_
        np.save(f'saved_urf_coords/fimp_hlabel{h}_{i}.npy', fimp)

np.save('saved_urf_coords/accu_hc.npy', accu)
