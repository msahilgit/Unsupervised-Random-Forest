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



trajs = [ 'sic1.xtc' ]

gap=3+1
total=90
rpairs = []
for i in range(0,total-gap):
    for j in range(i+gap,total):
        rpairs.append([i,j])
rpairs = np.array(rpairs).astype(int)

np.savetxt('rpairs_prconf.txt', rpairs, fmt='%d')


pdb=coor.featurizer('sic1.pdb')
pdb.add_residue_mindist(residue_pairs=rpairs, scheme='closest-heavy', periodic=False)
out = coor.source(trajs, features=pdb).get_output()

np.save('prconf_dists.npy', out)



