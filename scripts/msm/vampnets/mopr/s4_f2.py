import numpy as np
import random
import os
import pickle as pkl
import sklearn as skl
from sklearn.model_selection import train_test_split as splt
from sklearn.preprocessing import MinMaxScaler as mms, StandardScaler as sds
import vampnet
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, BatchNormalization, concatenate, Dropout
from keras import optimizers
import tensorflow as tf
import keras.backend as bkd
import tensorflow.keras.backend as tbkd



traj_data = np.load('../1_data/s229_dists.npz')
traj_data = [traj_data[i] for i in list(traj_data)]
randoms = np.loadtxt('../1_data/randoms.txt', dtype=int)

features = np.concatenate((traj_data))
scaler = sds()
scaler.fit(features)
traj_data = [scaler.transform(i) for i in traj_data]

nobs, input_size = features.shape
nodes = [128, 64, 32, 16, 8]
output_size = 4
factor = 2 #for saving files

drl = 2
dropouts = [0.1 if a < drl else 0 for a in range(len(nodes))]

learning_rate = 0.001
nb_epochs = [50, 100] #divided by 1:2 for vamp1 pre-training and vamp2 training
batch_size = 4096

epsilon = 1e-5
vamp = vampnet.VampnetTools(epsilon = epsilon)

lags = np.arange(5, 50+1, 5)
for lag in lags:
    f0 = np.concatenate(([i[:-lag] for i in traj_data]))
    f1 = np.concatenate(([i[lag:] for i in traj_data]))
    indices = np.arange(len(f0))

    for i in range(len(randoms)):

        train, test = splt(indices, test_size=0.3, random_state=randoms[i])

        #seeds
        random.seed(randoms[i])
        np.random.seed(randoms[i])
        tf.random.set_seed(randoms[i])
        os.environ['PYTHONHASHSEED'] = str(randoms[i])
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

        # === VAMPNET =====
        if 'model' in globals():
            del model
            bkd.clear_session()
            tbkd.clear_session()

        d0 = Input(shape=(input_size,))
        d1 = Input(shape=(input_size,))

        bn_layer = Activation('linear')
        dense_layers = [Dense(node, activation='relu') for node in nodes]
        dropout_layers = [Dropout(dropouts[a]) for a, nodes in enumerate(nodes)]

        l0 = bn_layer(d0)
        l1 = bn_layer(d1)
        for a, layer in enumerate(dense_layers):
            l0 = dense_layers[a](l0)
            l0 = dropout_layers[a](l0)
            l1 = dense_layers[a](l1)
            l1 = dropout_layers[a](l1)

        softmax = Dense(output_size, activation='softmax')
        l0 = softmax(l0)
        l1 = softmax(l1)

        merged = concatenate([l0, l1])

        model = Model(inputs=[d0,d1], outputs=merged)
        adam = optimizers.Adam(learning_rate = learning_rate)
        losses = [
            vamp._loss_VAMP_sym,
            vamp.loss_VAMP2
            ]
        valid_metric = []
        train_metric = []
        for l_index, loss in enumerate(losses):
            model.compile(optimizer = 'adam', loss = loss, metrics = [vamp.metric_VAMP])
            history = model.fit( [f0[train], f1[train]], np.zeros(( len(train), output_size*2)),
                                validation_data = ([f0[test], f1[test]], np.zeros(( len(test), output_size*2)) ),
                                batch_size=batch_size, epochs=nb_epochs[l_index] )

            valid_metric.append(history.history['val_metric_VAMP'])
            train_metric.append(history.history['metric_VAMP'])

        states_prob = model.predict([f0,f1])[:,:output_size]
        states_prob = np.argmax(states_prob, axis=1)

        model.save(f'saved_vampnet/model_s{output_size}_f{factor}_{lag}_{i}.h5')
        np.savez(f'saved_vampnet/train_metric_s{output_size}_f{factor}_{lag}_{i}.npz', *train_metric)
        np.savez(f'saved_vampnet/valid_metric_s{output_size}_f{factor}_{lag}_{i}.npz', *valid_metric)
        np.save(f'saved_dtrjs/dtrj_s{output_size}_f{factor}_{lag}_{i}.npy', states_prob)


