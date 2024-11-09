import numpy as np
np.bool = np.bool_
import pyemma.coordinates as coor
from sklearn.model_selection import train_test_split as splt
import sys
sys.path.append('../../../4_amino/999_0_practice/amino/reproducibility/')
import amino_original
sys.path.append('../../../0_python_modules/')
import navjeet_hist as nh


traj_data = np.load('../../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data)]
features = np.concatenate((traj_data))[::20]
dists = np.concatenate((  np.load('../../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

selected_ops = []
hists = []
extents = []
for i in range(len(randoms)):
    xtrain, xtest = splt(features, test_size=0.3, random_state=randoms[i])

    ops = [amino_original.OrderParameter(str(a), f) for a,f in enumerate(xtrain.T)]
    print(i, len(ops))
    fops = amino_original.find_ops( ops, max_outputs=40, bins=80, jump_filename=f'saved_original/distortion_jumps_{i}')

    sel = [int(op.name) for op in fops]
    tic2 = [ t[:,sel] for t in traj_data]
    tic2 = np.concatenate(( coor.tica(tic2, lag=10, dim=2).get_output() ))
    hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1)
    selected_ops.append(sel)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_original/hists.npy', hists)
np.save('saved_original/extents.npy', extents)
np.savez('saved_original/selected_ops.npz', *selected_ops)




