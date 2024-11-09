import numpy as np
np.bool = np.bool_
import pyemma.coordinates as coor
import sys
sys.path.append('../../../0_python_modules/')
import navjeet_hist as nh



traj_data = np.load('../../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data)]
dists = np.concatenate((  np.load('../../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

ntic2 = coor.tica_nystroem(max_columns=40, data=traj_data, lag=10, dim=2, nsel=1, initial_columns=1).column_indices

np.save('saved_mopr_nystroem/selected_columns.npy',ntic2)

ntic2 = [i[:,ntic2] for i in traj_data]
ntic2 = np.concatenate(( coor.tica(ntic2, lag=10, dim=2).get_output() ))

hh = nh.hist_range(ntic2, dists, mini=0.35, maxi=1)

np.save('saved_mopr_nystroem/hists.npy', hh[4])
np.save('saved_mopr_nystroem/extents.npy', hh[:4])





