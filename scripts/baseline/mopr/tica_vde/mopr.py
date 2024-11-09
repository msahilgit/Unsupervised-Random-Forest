from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np

import sys
sys.path.append('999_0_vde/vde/')
from vde import VDE

sys.path.append('../0_python_modules/')
import extras
import navjeet_hist as nh


tic2 = np.load('1_datas/tic2_mopr.npz')
tic2 = [tic2[i] for i in list(tic2)]
dists = np.concatenate(( np.load('1_datas/distance_alldata.npy', allow_pickle=True) ))

hidden_sizes = [32, 256]
hidden_depths = [2, 3]
input_size = tic2[0].shape[1]
lag_time = 10
encoder_size = 2
batch_size = 1024
scale = 1e-3
dropout = 0
lr=1e-4
optimizer='Adam'
n_epochs = 40
sliding_window = True
autocorr = True
cuda = False
verbose = True
loss = 'MSELoss'
activation = 'Swish'

hists = []
extents = []
for s in hidden_sizes:
    h1 = []
    e1 = []
    for d in hidden_depths:
        model = VDE(input_size=input_size, lag_time=lag_time, encoder_size=encoder_size,
                    hidden_layer_depth=d, hidden_size=s,
                    scale=scale, batch_size=batch_size, n_epochs=n_epochs, dropout_rate=dropout, learning_rate=lr,
                    optimizer=optimizer, activation=activation, loss=loss,
                    sliding_window=sliding_window, autocorr=autocorr, cuda=cuda, verbose=verbose
                    )

        model.fit(tic2)
        encoded = model.transform( [np.concatenate((tic2))] )[0]
        np.save(f'saved_mopr/encoded_{s}_{d}.npy', encoded)
        
        hh = nh.hist_range(encoded, dists, mini=0.35, maxi=1)
        h1.append(hh[4])
        e1.append(hh[:4])
    hists.append(h1)
    extents.append(e1)

np.save('saved_mopr/hists.npy', hists)
np.save('saved_mopr/extents.npy', extents)

