import copy as cp
import numpy as np
np.bool = np.bool_
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
from sklearn.model_selection import train_test_split as splt
from sklearn.preprocessing import MinMaxScaler as mms, StandardScaler as sds, Normalizer as nmlz
from sklearn.metrics import accuracy_score as acc, confusion_matrix as cfm
import pyemma
import pyemma.coordinates as coor
import random
import os
import sys
sys.path.append('../../../0_python_modules/')
import extras
import navjeet_hist as nh


traj_data = np.load('../../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data)]
dists = np.concatenate((  np.load('../../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

features = np.concatenate((traj_data))
scaler = sds()
scaler.fit(features)
features = scaler.transform(features[::20])

scaler = mms()
scaler.fit(dists[:,None])
labels = scaler.transform(dists[::20][:,None])[:,0]

def get_saliency(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
    saliency = tape.gradient(y_pred, x)
    return saliency

accu = np.zeros((len(randoms)))+np.nan
hists = []
extents = []
for i in range(len(randoms)):

    random.seed(randoms[i])
    np.random.seed(randoms[i])
    tf.random.set_seed(randoms[i])
    os.environ['PYTHONHASHSEED'] = str(randoms[i])
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    xtrain, xtest, ytrain, ytest = splt(features, labels, test_size=0.3, random_state=randoms[i])

    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(229,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(xtrain, ytrain, epochs=200, batch_size=128, validation_data=(xtest, ytest))

    pred = model.predict(xtest)
    accu[i] = np.sqrt(np.mean(np.square(ytest-pred)))

    for v in ['loss', 'val_loss']:
        np.save(f'saved_nn2/{v}_{i}.npy', history.history[v])
    model.save(f'saved_nn2/nn1_{i}.h5')

    xtf = tf.convert_to_tensor(features, dtype=tf.float32)
    fimp = get_saliency(model, xtf)
    fimp = tf.reduce_mean(fimp, axis=0)
    np.save(f'saved_nn2/fimp_{i}.npy', fimp)

    fimp = np.argsort(fimp)[::-1][:40]
    tic2 = [t[:,fimp] for t in traj_data]
    tic2 = np.concatenate(( coor.tica(tic2, lag=10, dim=2).get_output() ))
    hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_nn2/accu.npy', accu)
np.save('saved_nn2/hists.npy', hists)
np.save('saved_nn2/extents.npy', extents)






