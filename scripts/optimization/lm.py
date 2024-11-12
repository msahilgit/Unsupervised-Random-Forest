import numpy as np
import sys
import time
sys.path.append('../../0_python_modules/')
from model import unsupervised_random_forest as urf


features = np.load('../../1_datasets/t4l/features.npy')

psize=0.1
palg='lm'

dobj = urf(rf1_cv=5, 
        pmt_alg=palg, pmt_data_size=psize,
        rf2_cv=5,
        save_dtrj=f'saved_lm/dtrj_{psize}',
        n_jobs=96
        )
dobj.fit(features)

np.save(f'saved_lm/lc_{psize}.npy', dobj.get_output()[0])
np.save(f'saved_lm/fimp_{psize}.npy', dobj.get_output()[1])

time.sleep(10)

