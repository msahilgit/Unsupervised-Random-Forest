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
sys.path.append('../0_python_modules/')
import extras
import navjeet_hist as nh


features = np.load('../1_datasets/t4l/features.npy')
labels = np.load('../1_datasets/t4l/labels.npy').astype(int)
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)
traj_data = [np.load(f'../2_jctc_urf/traj_specific_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt('../2_jctc_urf/traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])

scaler = sds()
scaler.fit(features)
features = scaler.transform(features)

def get_saliency(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
    saliency = tape.gradient(y_pred, x)
    return saliency

accu = np.zeros((len(randoms),2))+np.nan
hists = []
extents = []
dropout=0.5
for i in range(len(randoms)):

    random.seed(randoms[i])
    np.random.seed(randoms[i])
    tf.random.set_seed(randoms[i])
    os.environ['PYTHONHASHSEED'] = str(randoms[i])
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    xtrain, xtest, ytrain, ytest = splt(features, labels, test_size=0.3, random_state=randoms[i])

    model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(755,)),
        layers.Dropout(dropout),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(32, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(16, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(8, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(4, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(2, activation='relu'),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
        ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(xtrain, ytrain, epochs=200, batch_size=512, validation_data=(xtest, ytest))

    pred = model.predict(xtest)
    pred = np.where(pred>0.5,1,0)[:,0]

    accu[i,0] = acc(ytest, pred)
    accu[i,1] = extras.get_f1_score( cfm(ytest, pred) )

    for v in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
        np.save(f'saved_d0.5_nn1/{v}_{i}.npy', history.history[v])
    model.save(f'saved_d0.5_nn1/nn1_{i}.h5')

    xtf = tf.convert_to_tensor(features, dtype=tf.float32)
    fimp = get_saliency(model, xtf)
    fimp = tf.reduce_mean(fimp, axis=0)
    np.save(f'saved_d0.5_nn1/fimp_{i}.npy', fimp)

    fimp = np.argsort(fimp)[::-1][:200]
    tic2 = [t[:,fimp] for t in traj_data]
    tic2 = coor.tica(tic2, lag=700, dim=2).get_output()[0]
    hh = nh.hist_range(tic2, dists, mini=0, maxi=0.6)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_d0.5_nn1/accu.npy', accu)
np.save('saved_d0.5_nn1/hists.npy', hists)
np.save('saved_d0.5_nn1/extents.npy', extents)






