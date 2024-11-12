import numpy as np
import sys
import time
sys.path.append('../../0_python_modules/')
from model import unsupervised_random_forest as urf


features = np.load('../../1_datasets/t4l/features.npy')

psize=1
palg='ef'

dobj = urf(pmt_alg=palg, pmt_data_size=psize,
        save_dtrj=f'saved_ef/dtrj_{psize}',
        n_jobs=96
        )
dobj.fit(features)

np.save(f'saved_ef/lc_{psize}.npy', dobj.get_output()[0])
np.save(f'saved_ef/fimp_{psize}.npy', dobj.get_output()[1])

time.sleep(10)

