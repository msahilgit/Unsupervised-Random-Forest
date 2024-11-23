import numpy as np
np.bool = np.bool_
import copy as cp
import matplotlib as mt
import matplotlib.pyplot as plt
import pyemma.plots as mplt
import pyemma.coordinates as coor
import pyemma
import sklearn as skl
from tqdm import tqdm



trajs = [
    f'LONG_TRAJ/{i}.xtc' for i in ['1f','2_2','2f_1','3_1','3_2','4f','5f']
] + [
    f'SHORT_TRAJ_subunit1/{i}.xtc' for i in range(1,63)
] + [
    f'SHORT_TRAJ_subunit2/{i}.xtc' for i in range(1,41)
] + [
    f'SHORT_TRAJ_intermediate/{i}.xtc' for i in range(1,21)
]


resi = np.arange(1,230)
rindi = []
for l in open('p2e.pdb', 'r'):
    i = l.strip().split()

    if i[0] == 'ATOM':

        if i[3] == 'LIG':
            if i[2] == 'CZ':
                lig=int(i[1])

        else:
            if i[2] == 'CA':
                if int(i[5]) in resi:
                    rindi.append(int(i[1]))

rpairs = np.zeros(( len(rindi), 2 ))
rpairs[:,0], rpairs[:,1] = rindi, lig
rpairs -= 1


pdb=coor.featurizer('p2e.pdb')
pdb.add_distances(indices=rpairs.astype(int))
out = coor.source(trajs, features=pdb).get_output()


out2 = []
for i in out:
    out2.append(np.minimum(i[:,:229], i[:,229:]))

np.savez('s229_dists.npz', *out2)



