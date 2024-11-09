import numpy as np
np.bool = np.bool_
import pyemma.coordinates as coor
import sys
sys.path.append('../0_python_modules/')
import navjeet_hist as nh


import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'


traj_data = [np.load(f'../2_jctc_urf/traj_specific_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt('../2_jctc_urf/traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])


#ntic2 = coor.tica_nystroem(max_columns=200, data=traj_data, lag=700, dim=2, nsel=1, initial_columns=1).column_indices

#np.save('saved_t4l_nystroem/selected_columns.npy',ntic2)
ntic2 = np.load('saved_t4l_nystroem/selected_columns.npy').astype(int)

ntic2 = [i[:,ntic2] for i in traj_data]
ntic2 = coor.tica(ntic2, lag=700, dim=2).get_output()[0]

hh = nh.hist_range(ntic2, dists, mini=0, maxi=0.6)

np.save('saved_t4l_nystroem/hists.npy', hh[4])
np.save('saved_t4l_nystroem/extents.npy', hh[:4])





