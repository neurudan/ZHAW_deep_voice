import pickle

import numpy as np


import common.spectrogram.speaker_train_splitter as sts
from .core import plot_saver as ps

np.random.seed(1337)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.layers import Layer
import tensorflow as tf
from .core import data_gen as dg
from .core import pairwise_kl_divergence as kld
from common.utils.load_config import *

from common.utils.paths import *

'''This Class Trains a Bidirectional LSTM with 2 Layers, and 2 Denselayer and a Dropout Layers
    Parameters:
    name: Network/experiment name to save artifacts
    training_data: name of the Training data file, expected to be train_xxxx.pickle
    n_hidden1: Units of the first LSTM Layer
    n_hidden2: Units of the second LSTM Layer
    n_classes: Amount of output classes (Speakers in Trainingset)
    n_10_batches: Number of Minibatches to train the Network (1 = 10 Minibatches)
    segment_size: Segment size that is used as input 100 equals 1 second with current Spectrogram extraction
    frequency: size of the frequency Dimension of the Input Spectrogram

    Work of Gerber and Glinski.
'''

class CosFace(Layer):
    def __init__(self, n_classes=100, regularizer=None, **kwargs):
        super(CosFace, self).__init__(**kwargs)
        config = load_config(None, join(get_common(), 'config.cfg'))
        self.n_classes = config.getint('pairwise_lstm_cosface', 'n_classes')
        self.regularizer = regularizer

    def build(self, input_shape):
        super(CosFace, self).build(input_shape[0])

        print(input_shape)
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        x = tf.nn.l2_normalize(inputs, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)

        logits = x @ W
        return logits

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

class bilstm_2layer_dropout(object):
    def __init__(self, name, training_data, n_hidden1, n_hidden2, n_classes, n_10_batches,
                 segment_size, loss, frequency=128):
        self.network_name = name
        self.training_data = training_data
        self.test_data = 'test' + training_data[5:]
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_classes = n_classes
        self.n_10_batches = n_10_batches
        self.segment_size = segment_size
        self.loss = loss
        self.input = (segment_size, frequency)
        print(self.network_name)
        self.run_network()

    def create_net(self):
        config = load_config(None, join(get_common(), 'config.cfg'))
        model = Sequential()
        model.add(Bidirectional(LSTM(self.n_hidden1, return_sequences=True), input_shape=self.input))
        model.add(Dropout(0.50))
        model.add(Bidirectional(LSTM(self.n_hidden2)))
        model.add(Dense(self.n_classes * 10))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_classes * 5))
        model.add(CosFace(self.n_classes))
        adam = keras.optimizers.Adam(config.getfloat('pairwise_lstm_cosface', 'adam_lr'),
                                     config.getfloat('pairwise_lstm_cosface', 'adam_beta_1'),
                                     config.getfloat('pairwise_lstm_cosface', 'adam_beta_2'),
                                     config.getfloat('pairwise_lstm_cosface', 'adam_epsilon'),
                                     config.getfloat('pairwise_lstm_cosface', 'adam_decay'))
        # ada = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        model.compile(loss=self.loss,
                      optimizer=adam,
                      metrics=['accuracy'])
        return model


    def create_train_data(self):
        with open(get_speaker_pickle(self.training_data), 'rb') as f:
            (X, y, speaker_names) = pickle.load(f)

            splitter = sts.SpeakerTrainSplit(0.2, 10)
            X_t, X_v, y_t, y_v = splitter(X, y)

        return X_t, y_t, X_v, y_v

    def create_callbacks(self):
        csv_logger = keras.callbacks.CSVLogger(get_experiment_logs(self.network_name + '.csv'))
        net_saver = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_best.h5"),
            monitor='val_loss', verbose=1, save_best_only=True)
        net_checkpoint = keras.callbacks.ModelCheckpoint(
            get_experiment_nets(self.network_name + "_{epoch:05d}.h5"), period=100)
        return [csv_logger, net_saver, net_checkpoint]

    def run_network(self):
        model = self.create_net()
        calls = self.create_callbacks()

        X_t, y_t, X_v, y_v = self.create_train_data()
        train_gen = dg.batch_generator_lstm(X_t, y_t, 100, segment_size=self.segment_size)
        val_gen = dg.batch_generator_lstm(X_v, y_v, 100, segment_size=self.segment_size)
        # batches_t = ((X_t.shape[0] + 128 - 1) // 128)
        # batches_v = ((X_v.shape[0] + 128 - 1) // 128)

        history = model.fit_generator(train_gen, steps_per_epoch=10, epochs=self.n_10_batches,
                                      verbose=2, callbacks=calls, validation_data=val_gen,
                                      validation_steps=2, class_weight=None, max_q_size=10,
                                      nb_worker=1, pickle_safe=False)
        ps.save_accuracy_plot(history, self.network_name)
        ps.save_loss_plot(history, self.network_name)
        print("saving model")
        model.save(get_experiment_nets(self.network_name + ".h5"))
        # print "evaluating model"
        # da.calculate_test_acccuracies(self.network_name, self.test_data, True, True, True, segment_size=self.segment_size)
