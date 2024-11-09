import numpy as np
np.bool = np.bool_
import pyemma.coordinates as coor
import sys
sys.path.append('../../../../0_python_modules/')
import extras
import metrics
import navjeet_hist as nh


traj_data = np.load('../../../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]
randoms = np.loadtxt('../../../../1_datasets/randoms.txt', dtype=int)
dists = np.concatenate(( np.load('../../../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))

hists=[]
extents = []
for i in range(len(randoms)):
    fimp = np.load(f'saved_shap/shap_values_supervised{i}.npy')
    fimp = np.mean( np.abs(fimp), axis=0).argsort()[::-1][:40]
    tic2 = [i[:,fimp] for i in traj_data]
    tic2 = np.concatenate(( coor.tica(tic2, lag=10, dim=2).get_output() ))
    hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_shap/hists.npy', hists)
np.save('saved_shap/extents.npy', extents)


