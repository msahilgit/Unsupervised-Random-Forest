import numpy as np
np.bool = np.bool_
import random
import copy as cp
import sys
import os
import matplotlib as mt
import matplotlib.pyplot as plt
import pickle as pkl
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier as rfc, RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as splt
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
from sklearn.preprocessing import MinMaxScaler as mms, StandardScaler as sds, Normalizer as nmlz
import sklearn.metrics as mtr
import scipy as sc
import scipy.cluster as scc
from tqdm import tqdm
import time
import pyemma
import pyemma.coordinates as coor
import pyemma.plots as mplt

from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras import Input, Model
import tensorflow as tf
import keras.backend as bkd

sys.path.append('../../0_python_modules/')

import proximity_matrix as pmt
import synthetic_data as syn
import extras
from navjeet_hist import hist_raw_data
import navjeet_hist as nh
import metrics



traj_data = [np.load(f'../../2_jctc_urf/traj_specific_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt(f'../../2_jctc_urf/traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])
randoms = np.loadtxt('../../1_datasets/randoms.txt', dtype=int)

sizes = [len(i) for i in traj_data]
def resize(data, sizes):
    tot=0
    tdata=[]
    for i in sizes:
        tdata.append(data[tot:tot+i])
        tot = tot+i
    return tdata

features = np.concatenate((traj_data))
scaler = mms()
scaler.fit(features )
features = scaler.transform(features)

actv = 'tanh'
kini = 'glorot_uniform'

hists = []
extents = []
for i in range(len(randoms)):

    # seeds
    random.seed(randoms[i])
    np.random.seed(randoms[i])
    tf.random.set_seed(randoms[i])
    os.environ['PYTHONHASHSEED'] = str(randoms[i])
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


    xtrain, xtest = splt(features, test_size=0.3, random_state=randoms[i])


    #model
    inp = Input(shape=(755,))
    e1 = Dense(400, activation=actv, kernel_initializer=kini)(inp)
    en = Dense(200, activation=actv, kernel_initializer=kini)(e1)
    d1 = Dense(400, activation=actv, kernel_initializer=kini)(en)
    out = Dense(755, activation=actv, kernel_initializer=kini)(d1)

    
    ae = Model(inp,out)
    encoder = Model(inp,en)

    ae.compile(optimizer='adam', loss='mae')
    ae.summary()

    tae = ae.fit( xtrain, xtrain, epochs=200, validation_data=(xtest, xtest), batch_size=512 )

    ae.save(f'saved_tanh_ae3/ae_{i}.h5')
    encoder.save(f'saved_tanh_ae3/encoder_{i}.h5')
    np.save(f'saved_tanh_ae3/loss_{i}.npy', tae.history['loss'])
    np.save(f'saved_tanh_ae3/val_loss_{i}.npy', tae.history['val_loss'])


    encoded = encoder.predict(features)
    encoded = resize(encoded, sizes)
    encoded = coor.tica( encoded, lag=700, dim=2).get_output()[0]

    hh = nh.hist_range(encoded, dists, mini=0, maxi=0.6 )
    hists.append(hh[4])
    extents.append(hh[:4])

    bkd.clear_session()

np.save('saved_tanh_ae3/hists.npy', hists)
np.save('saved_tanh_ae3/extents.npy', extents)



