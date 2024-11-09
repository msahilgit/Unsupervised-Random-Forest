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
import scipy.cluster as scc
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
features = np.concatenate((traj_data))[::5]

randoms = np.loadtxt('../1_datasets/randoms.txt').astype(int)


create_permute = syn.synthetic_data(features, size=1)
create_permute.permute()
pfeatures, plabels = create_permute.get_output()

clfs = []
for i in range(len(randoms)):
    xtrain, _, ytrain, _ = splt(pfeatures, plabels, test_size=0.3, random_state=randoms[i])
    clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    clf_.fit(xtrain, ytrain)

    xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
    clfs.append(clf_)
    pkl.dump(clf_, open(f'saved_furf/clf_permute{i}.pkl','wb'))


permute_pmt = pmt.calculate_condenced_pmt_multi( clfs, features, n_jobs=94 )
np.save('saved_furf/permute_pmt_mean.npy', permute_pmt)


hc_permute = scc.hierarchy.linkage(permute_pmt, method='average')
np.save('saved_furf/hc_permute.npy', hc_permute)
hc_permute = np.load('saved_furf/hc_permute.npy')


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
        pkl.dump(clf_, open(f'saved_furf/clf_hlabel{h}_{i}.pkl','wb'))
        fimp = clf_.feature_importances_
        np.save(f'saved_furf/fimp_hlabel{h}_{i}.npy', fimp)


np.save('saved_furf/accu_hc.npy', accu)






