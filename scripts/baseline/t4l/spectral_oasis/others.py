import numpy as np
np.bool = np.bool_
import pyemma.coordinates as coor
import sys
sys.path.append('../0_python_modules/')
import navjeet_hist as nh




traj_data = [np.load(f'../2_jctc_urf/traj_specific_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt('../2_jctc_urf/traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])

initial_columns = [10, 30, 50]
nsel = [3, 5, 10]

hists=[]
extents=[]
selected_cols=[]
for ini in initial_columns:
    h1=[]
    e1=[]
    s1=[]
    for n in nsel:
        ntic2 = coor.tica_nystroem(max_columns=200, data=traj_data, lag=700, dim=2, nsel=n, initial_columns=ini).column_indices
        s1.append(ntic2)
        ntic2 = [i[:,ntic2] for i in traj_data]
        ntic2 = coor.tica(ntic2, lag=700, dim=2).get_output()[0]
        hh = nh.hist_range(ntic2, dists, mini=0, maxi=0.6)
        h1.append(hh[4])
        e1.append(hh[:4])
    hists.append(h1)
    extents.append(e1)
    selected_cols.append(s1)

np.save(f'saved_others/hists.npy', hists)
np.save(f'saved_others/extents.npy', extents)
np.save(f'saved_others/selected_cols.npy', selected_cols)





