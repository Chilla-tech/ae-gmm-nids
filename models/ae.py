# Autoencoder model definition for AE + GMM NIDS
# Builds a feedforward neural network autoencoder with L1 regularization.

import numpy as np
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def build_autoencoder(input_dim: int, l1=1e-5, lr=1e-3, clipN=1.0):
    inp = Input(shape=(input_dim,))
    enc = Dense(90, activation='relu', activity_regularizer=regularizers.l1(l1))(inp)
    enc = Dense(70, activation='relu', activity_regularizer=regularizers.l1(l1))(enc)
    enc = Dense(30, activation='relu', activity_regularizer=regularizers.l1(l1))(enc)
    enc = Dense(17, activation='relu', activity_regularizer=regularizers.l1(l1))(enc)
    enc = Dense(16, name='latent_space', activation='relu', activity_regularizer=regularizers.l1(l1))(enc)

    dec = Dense(17, activation='relu', activity_regularizer=regularizers.l1(l1))(enc)
    dec = Dense(30, activation='relu', activity_regularizer=regularizers.l1(l1))(dec)
    dec = Dense(70, activation='relu', activity_regularizer=regularizers.l1(l1))(dec)
    dec = Dense(90, activation='relu', activity_regularizer=regularizers.l1(l1))(dec)
    out = Dense(input_dim, activation='linear')(dec)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mae', optimizer=Adam(learning_rate=lr, clipnorm=clipN))
    #model.summary()
    return model