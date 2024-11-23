import copy as cp
import numpy as np
np.bool = np.bool_
np.float = np.float_
import pyemma.coordinates as coor



trajs = [f'pbc_trajs/{i:03d}.xtc' for i in range(200)]

rpairs = np.array([ [i,j] for i in range(32) for j in range(i+1,32)]).astype(int)


pdb=coor.featurizer('ref.pdb')
pdb.add_residue_mindist(residue_pairs=rpairs, scheme='closest-heavy', periodic=False)
out = coor.source(trajs, features=pdb).get_output()


np.savez('alldists.npz', *out)












