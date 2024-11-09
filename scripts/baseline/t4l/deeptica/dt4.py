import copy as cp
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as mms
import sys
sys.path.append('../../0_python_modules/')
import extras
import metrics
import navjeet_hist as nh
sys.path.append('../deep-learning-slow-modes/')
import mlcvs
from mlcvs.io import load_colvar
from mlcvs.tica import DeepTICA,create_tica_dataset
from mlcvs.palette import cm_fessa
from mlcvs.fes import compute_fes_1d, compute_fes_2d


traj_data = [ np.load(f'../1_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt('../1_data/distances0.xvg', comments=['@','#'], usecols=[3])
lag=700
times = []
add=-lag-3
for a,i in enumerate(traj_data):
    tt = np.arange(len(i)) + (lag+3) + add
    times.append(tt)
    add = tt[-1]

features = np.concatenate((traj_data))
traj_data = traj_data[0]
times = np.concatenate((times))

scaler = mms()
scaler.fit(features)
features = scaler.transform(features)
traj_data = scaler.transform(traj_data)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

nobs = features.shape[0]
n_train = int(nobs*0.7)
n_valid = nobs-n_train

train, test = create_tica_dataset(features, times, lag_time=lag, n_train=n_train, n_valid=n_valid)

# hyperparameters
nodes = [features.shape[1], 32, 32, 4]
activ_type = 'tanh'
loss_type = 'sum2'
n_eig = 2
num_epochs = 200
log_every = 30
lrate = 1e-3
l2_reg = 0
earlystop = True
es_patience = 10
es_consecutive = True

model = DeepTICA(nodes)

opt = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=0)

model.set_optimizer(opt)
model.set_earlystopping(patience=es_patience, min_delta=0.001, consecutive=True, save_best_model=True, log=False)

model.train(train, test,
            standardize_inputs=True,
            standardize_outputs=True,
            loss_type=loss_type,
            n_eig=n_eig,
            nepochs=num_epochs,
            info=True,log_every=log_every)

# save model
torch.save(model.state_dict(), 'saved_dt4/model_state_dict.pth')  #model.load_state_dict(torch.load('saved_dt4/model_state_dict.pth'))
torch.save(opt.state_dict(), 'saved_dt4/optimizer_state_dict.pth') #opt.load_state_dict(torch.load('saved_dt4/optimizer_state_dict.pth'))
#
np.save(f'saved_dt4/loss_train.npy', model.loss_train)
np.save(f'saved_dt4/loss_valid.npy', model.loss_valid)
np.save(f'saved_dt4/evals_train.npy', np.asarray(torch.cat(model.evals_train)) )

with torch.no_grad():
    dtic2 = model(torch.Tensor(traj_data)).numpy()
np.save(f'saved_dt4/dtic2.npy', dtic2)

hh = nh.hist_range(dtic2, dists, mini=0, maxi=0.6)
np.save('saved_dt4/hists.npy', hh[4])
np.save('saved_dt4/extents.npy', hh[:4])



