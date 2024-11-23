import copy as cp
import numpy as np
np.bool = np.bool_
np.float = np.float_
import pyemma.coordinates as coor



trajs = [f'trajs_from_bhuppi/no_pbc2_{i}.xtc' for i in range(12,21)]


coords = []
for l in open('trajs_from_bhuppi/ref.pdb','r'):
    i=l.strip().split()
    if i[0] == 'ATOM':
        if i[2] =='CA' or i[2] == 'FE':
            coords.append(i[5:8])
coords = np.array(coords).astype(float)


cutt = 10
gap = 4
rpairs = []
for i in range(len(coords)-gap):
    for j in range(i+gap, len(coords)):

        if np.linalg.norm(coords[i]-coords[j]) <= cutt:
            rpairs.append([i,j])

rpairs = np.array(rpairs).astype(int)


np.savetxt('rpairs_prconf.txt', rpairs, fmt='%d')




pdb=coor.featurizer('trajs_from_bhuppi/ref.pdb')
pdb.add_residue_mindist(residue_pairs=rpairs, scheme='closest-heavy', periodic=False)
out = coor.source(trajs, features=pdb).get_output()


np.savez('prconf_dists.npz', *out)












