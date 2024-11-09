import numpy as np
np.bool = np.bool_
import mosaic as msc
import pyemma.coordinates as coor
from sklearn.model_selection import train_test_split as splt
import sys
sys.path.append('../0_python_modules/')
import navjeet_hist as nh




features=np.load('../1_datasets/t4l/features.npy')
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)
traj_data = [np.load(f'../2_jctc_urf/traj_specific_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt('../2_jctc_urf/traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])


hists = []     
extents = []
for i in range(len(randoms)):
    xtrain, _ = splt(features, test_size=0.3, random_state=randoms[i])

    sim = msc.Similarity(metric='correlation')
    sim.fit(xtrain)
    cmat = sim.matrix_
    clus = msc.Clustering(mode='CPM', weighted=True, resolution_parameter=0.5, seed=int(randoms[i]))
    clus.fit(cmat)
    np.save(f'saved_t4l_mosaic/clusters_{i}.npy', clus.clusters_)

    sizes = np.array([len(c) for c in clus.clusters_])
    indices = np.argsort(sizes)[::-1]
    clusters = clus.clusters_[indices]
    sizes = sizes[indices]

    fimp1 = np.concatenate((clusters[:1]))
    fimp3 = np.concatenate((clusters[:3]))
    if np.max(sizes)>=5: 
        last5 = np.where(sizes>=5)[0][-1] ; do5=True
        fimp5 = np.concatenate((clusters[:last5+1]))
    else: 
        do5=False
        fimp5=None
    dos = [True, True, do5, True]
    fimp200 = np.concatenate((clusters))[:200]

    h1=[]
    e1=[]
    for a, fimp in enumerate([fimp1, fimp3, fimp5, fimp200]):
        if dos[a]:
            tic2 = [t[:,fimp] for t in traj_data]
            tic2 = coor.tica(tic2, lag=700, dim=2).get_output()[0]
            hh = nh.hist_range(tic2, dists, mini=0, maxi=0.6)
            h1.append(hh[4])
            e1.append(hh[:4])
        else:
            h1.append(np.zeros((100,100))+np.nan)
            e1.append(np.zeros((4))+np.nan)
    hists.append(h1)
    extents.append(e1)

np.save('saved_t4l_mosaic/hists.npy', hists)
np.save('saved_t4l_mosaic/extents.npy', extents)


