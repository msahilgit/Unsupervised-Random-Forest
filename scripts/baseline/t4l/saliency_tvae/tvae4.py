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
sys.path.append('../0_python_modules/')
import extras
import navjeet_hist as nh

#---------------------------------------------------------------

traj_data = [np.load(f'../2_jctc_urf/traj_specific_data/features_{i}.npy') for i in range(6)]
dists = np.loadtxt('../2_jctc_urf/traj_specific_data/distances0.xvg', comments=['@','#'], usecols=[3])
randoms = np.loadtxt('../1_datasets/randoms.txt', dtype=int)

lag=700
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

inp = Input(shape=(755,))
e1 = Dense(64, activation='relu')(inp)
e2 = Dense(32, activation='relu')(e1)

means = Dense(16, activation='relu')(e2)
stds = Dense(16, activation='relu')(e2)
z = sampling()([means,stds])

encoder = Model(inp, [means, stds, z])
encoder.summary()

en = Input(shape=(16,))
d1 = Dense(32, activation='relu')(en)
d2 = Dense(64, activation='relu')(d1)
out = Dense(755, activation='sigmoid')(d2)

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
    np.save(f'saved_tvae4/rc_loss_{i}.npy', history.history['reconstruction_loss'])
    np.save(f'saved_tvae4/kl_loss_{i}.npy', history.history['kl_loss'])

    fimp = get_saliency(model, start, end)
    np.save(f'saved_tvae4/saliency_{i}.npy', fimp)

    bkd.clear_session()
    tbkd.clear_session()
    
    fimp = fimp.argsort()[::-1][:200]
    tic2 = [i[:,fimp] for i in traj_data]
    tic2 = coor.tica(tic2, dim=2, lag=lag).get_output()[0]
    hh = nh.hist_range(tic2, dists, mini=0, maxi=0.6)
    hists.append(hh[4])
    extents.append(hh[:4])

np.save('saved_tvae4/hists.npy', hists)
np.save('saved_tvae4/extents.npy', extents)
    
