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
traj_data = [traj_data[i] for i in list(traj_data)]
dists = np.concatenate(( np.load('../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))

#features = np.concatenate(( traj_data ))[::20]
randoms = np.loadtxt('../../1_datasets/randoms.txt', dtype=int)

#create_permute = syn.synthetic_data(features, size=1)
#create_permute.permute()
#pfeatures, plabels = create_permute.get_output()

#np.save('saved_furf/pfeatures.npy', pfeatures)
#np.save('saved_furf/plabels.npy', plabels)

#clfs = []
#accu = np.zeros((len(randoms),2))
#for i in range(len(randoms)):
    #xtrain, xtest, ytrain, ytest = splt(pfeatures, plabels, test_size=0.3, random_state=randoms[i])
    #clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
    #clf_.fit(xtrain, ytrain)

    #pred_ = clf_.predict(xtest)
    #accu[i,0] = acc(ytest,pred_)
    #accu[i,1] = extras.get_f1_score( cfm(ytest, pred_) )


    #xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
    #clfs.append(clf_)
    #pkl.dump(clf_, open(f'saved_furf/clf_permute{i}.pkl','wb'))

#np.save('saved_furf/accu_permute.npy', accu)


#permute_pmt = pmt.calculate_condenced_pmt_multi( clfs, features, n_jobs=94 )
#np.save('saved_furf/permute_pmt_mean.npy', permute_pmt)


#hc_permute = scc.hierarchy.linkage(permute_pmt, method='average')
#np.save('saved_furf/hc_permute.npy', hc_permute)




#accu = np.zeros((9,len(randoms),2))
for h in range(2,11):
    #hlabels_ = extras.get_hc_dtraj(hc_permute, nids=h)

    for i in range(len(randoms)):
        #xtrain, xtest, ytrain, ytest = splt(features, hlabels_, test_size=0.3, random_state=randoms[i])
        #clf_ = rfc(n_estimators=1000, random_state=randoms[i], n_jobs=94)
        #clf_.fit(xtrain, ytrain)

        #pred_ = clf_.predict(xtest)
        #accu[h-2,i,0] = acc(ytest,pred_)
        #accu[h-2,i,1] = extras.get_f1_score( cfm(ytest, pred_) )

        #xtrain, xtest, ytrain, ytest = 0, 0, 0, 0
        #pkl.dump(clf_, open(f'saved_furf/clf_hlabel{h}_{i}.pkl','wb'))
        #fimp = clf_.feature_importances_
        #np.save(f'saved_furf/fimp_hlabel{h}_{i}.npy', fimp)

        fimp = np.load(f'saved_furf/fimp_hlabel{h}_{i}.npy')
        fimp = fimp.argsort()[::-1][:40]
        tic2 = [ i[:,fimp] for i in traj_data]
        tic2 = np.concatenate(( coor.tica(tic2, lag=10, dim=2).get_output() ))

        #hh = nh.hist_range( tic2, dists, mini=0.35, maxi=1)
        #np.save(f'saved_furf/hist_{h}_{i}.npy', hh[4])
        #np.save(f'saved_furf/extents_{h}_{i}.npy', hh[:4])

        #hh = nh.hist_range( tic2, dists, mini=0, maxi=0.35)
        #np.save(f'saved_furf/different_state_bound/hist_{h}_{i}.npy', hh[4])
        #np.save(f'saved_furf/different_state_bound/extents_{h}_{i}.npy', hh[:4])

        hh = nh.hist_range( tic2, dists, mini=1, maxi=10)
        np.save(f'saved_furf/different_state_unbound/hist_{h}_{i}.npy', hh[4])
        np.save(f'saved_furf/different_state_unbound/extents_{h}_{i}.npy', hh[:4])


#np.save('saved_furf/accu_hc.npy', accu)



