import numpy as np
np.bool = np.bool_
import pyemma.coordinates as coor
import random
import pickle as pkl
import sklearn as skl
from sklearn.model_selection import train_test_split as splt
from sklearn.preprocessing import Normalizer as nmlr, MinMaxScaler as mms, StandardScaler as sds
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow as tf
from tensorflow.python.framework import ops
from keras import Input, Model
from keras.layers import Layer, Dense
import tensorflow.keras.backend as tbkd
import keras.backend as bkd
import sys
sys.path.append('../../../0_python_modules/')
import extras
import navjeet_hist as nh

#---------------------------------------------------------------

traj_data = np.load('../../../1_datasets/mopR_bindings/s229_dists.npz')
traj_data = [ traj_data[i] for i in list(traj_data)]
dists = np.concatenate((  np.load('../../../1_datasets/mopR_bindings/distance_alldata.npy', allow_pickle=True) ))
randoms = np.loadtxt('../../../1_datasets/randoms.txt', dtype=int)

lag=10
start = np.concatenate(([i[:-lag] for i in traj_data]))
end = np.concatenate(([i[lag:] for i in traj_data]))
indices = np.arange(len(start))

scaler = nmlr()
scaler.fit( np.concatenate((traj_data)) )
start = scaler.transform(start)
end = scaler.transform(end)

#---------------------------------------------------------------

class sampling(Layer):
    def call(self, inputs, seed=42, mn=0, std=1, scale=1e-3):
        mean, var = inputs
        eps = tf.random.normal(shape=tf.shape(mean), mean=mn, stddev=std, seed=seed)
        return mean + scale*eps*tf.exp(0.5*var)

inp = Input(shape=(229,))
e1 = Dense(64, activation='relu')(inp)

means = Dense(16, activation='relu')(e1)
stds = Dense(16, activation='relu')(e1)
z = sampling()([means,stds])

encoder = Model(inp, [means, stds, z])
encoder.summary()

en = Input(shape=(16,))
d1 = Dense(64, activation='relu')(en)
out = Dense(229, activation='sigmoid')(d1)

decoder = Model(en, out)
decoder.summary()


class vae(Model):

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = tf.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = tf.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        start, end = data
        with tf.GradientTape() as tape:
            mean, stds, z = self.encoder(start)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        keras.losses.mean_absolute_error(end, reconstruction),
                        )
                    )
            kl_loss = -0.5 * (1 + stds - tf.square(mean) - tf.exp(stds))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = kl_loss + reconstruction_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
                }

def get_saliency(model, start, end):
    xstart = tf.convert_to_tensor(start)
    xend = tf.convert_to_tensor(end)
    with tf.GradientTape() as tape:
        tape.watch(xstart)
        enc = model.encoder(xstart)[2]
        pred = model.decoder(enc)
        loss = tf.reduce_mean( tf.square(pred-xend), axis=1 )
        saliency = tf.reduce_mean(tape.gradient(loss, xstart), axis=0).numpy()
    return saliency
#---------------------------------------------------------------

hists=[]
extents=[]
for i in range(len(randoms)):

    xtrain, xtest = splt(indices, random_state=randoms[i], test_size=0.3)

    random.seed(randoms[i])
    np.random.seed(randoms[i])
    tf.random.set_seed(randoms[i])
    os.environ['PYTHONHASHSEED'] = str(randoms[i])
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    if 'model' in globals():
        del model

    model = vae(encoder, decoder)
    model.compile(optimizer=keras.optimizers.Adam())
    history = model.fit( start[xtrain][::10], end[xtrain][::10], 
            epochs=40, batch_size=32,
            )
    np.save(f'saved_tvae5/rc_loss_{i}.npy', history.history['reconstruction_loss'])
    np.save(f'saved_tvae5/kl_loss_{i}.npy', history.history['kl_loss'])

    fimp = get_saliency(model, start, end)
    np.save(f'saved_tvae5/saliency_{i}.npy', fimp)

    bkd.clear_session()
    tbkd.clear_session()
    
    fimp = fimp.argsort()[::-1][:40]
    tic2 = [i[:,fimp] for i in traj_data]
    tic2 = np.concatenate(( coor.tica(tic2, dim=2, lag=lag).get_output() ))
    hh = nh.hist_range(tic2, dists, mini=0.35, maxi=1)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_tvae5/hists.npy', hists)
np.save('saved_tvae5/extents.npy', extents)
    
