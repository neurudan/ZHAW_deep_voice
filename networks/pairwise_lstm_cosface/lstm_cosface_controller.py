"""
The controller to train and test the pairwise_lstm_cosface network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import keras
from keras import backend as K
import os

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils import TimeCalculator
from common.utils.ShortUtteranceConverter import create_data_lists
from common.utils.logger import *
from common.utils.paths import *
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout, CosFace
from .core.data_gen import generate_test_data
from .core.pairwise_kl_divergence import pairwise_kl_divergence
from common.spectrogram.speaker_dev_selector import load_test_data

class LSTMCosFaceController(NetworkController):
    def __init__(self, config):
        super().__init__("pairwise_lstm_"+config.get('pairwise_lstm_cosface', 'loss_func')+"_" + str(config.getint('pairwise_lstm_cosface', 'n_classes')) + "_m_" + str(config.getfloat('pairwise_lstm_cosface', 'margin')) + "_s_" + str(config.getfloat('pairwise_lstm_cosface', 'scale')), config)
        self.network_file = self.name
        self.loss_func = config.get('pairwise_lstm_cosface', 'loss_func')
        self.scale = config.getfloat('pairwise_lstm_cosface', 'scale')
        self.margin = config.getfloat('pairwise_lstm_cosface', 'margin')
        os.makedirs(get_experiments()+"/m_%08.5f_s_%06.2f" % (self.margin, self.scale), exist_ok=True)
        os.makedirs(get_experiments()+"/plots", exist_ok=True)
        os.makedirs(get_experiments()+"/logs", exist_ok=True)
        os.makedirs(get_experiments()+"/nets", exist_ok=True)
        os.makedirs(get_experiments()+"/results", exist_ok=True)
        self.loss = self.cosface_loss
        if self.loss_func == 'arcface':
            self.loss = self.arcface_loss

    def train_network(self):
        bilstm_2layer_dropout(
            self.network_file,
            self.config.get('train', 'pickle'),
            self.config.getint('pairwise_lstm_cosface', 'n_hidden1'),
            self.config.getint('pairwise_lstm_cosface', 'n_hidden2'),
            self.config.getint('pairwise_lstm_cosface', 'n_classes'),
            self.config.getint('pairwise_lstm_cosface', 'n_10_batches'),
            self.config.getint('pairwise_lstm_cosface', 'seg_size'),
            self.loss
        )

    def cosface_loss(self, y_true, y_pred):
        target_logits = y_pred - self.margin
        logits = y_pred * (1 - y_true) + target_logits * y_true
        logits *= self.scale
        out = tf.nn.softmax(logits)
        loss = keras.losses.categorical_crossentropy(y_true, out)
        return loss

    def arcface_loss(self, y_true, y_pred):
        theta = tf.acos(K.clip(y_pred, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.margin)
        logits = y_pred * (1 - y_true) + target_logits * y_true
        logits *= self.scale
        out = tf.nn.softmax(logits)
        loss = keras.losses.categorical_crossentropy(y_true, out)
        return loss

    def get_embeddings(self):
        short_utterance = self.config.getboolean('validation', 'short_utterances')
        out_layer = self.config.getint('pairwise_lstm_cosface', 'out_layer')
        seg_size = self.config.getint('pairwise_lstm_cosface', 'seg_size')
        vec_size = self.config.getint('pairwise_lstm_cosface', 'vec_size')

        logger = get_logger('lstm', logging.INFO)
        logger.info('Run pairwise_lstm_cosface test')
        logger.info('out_layer -> ' + str(out_layer))
        logger.info('seg_size -> ' + str(seg_size))
        logger.info('vec_size -> ' + str(vec_size))

        # Load and prepare train/test data
        x_train, speakers_train, s_list_train = load_test_data(self.get_validation_train_data())
        x_test, speakers_test, s_list_test = load_test_data(self.get_validation_test_data())
        x_train, speakers_train, = prepare_data(x_train, speakers_train, seg_size)
        x_test, speakers_test = prepare_data(x_test, speakers_test, seg_size)

        x_list, y_list, s_list = create_data_lists(short_utterance, x_train, x_test,
                                                   speakers_train, speakers_test, s_list_train, s_list_test)

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []
        set_of_total_times = []
        checkpoints = list_all_files(get_experiment_nets(), "^pairwise_lstm_cosface.*\.h5")

        # Values out of the loop
        metrics = ['accuracy', 'categorical_accuracy', ]
        custom_objects = {'CosFace': CosFace, 'cosface_loss': self.cosface_loss, 'arcface_loss': self.arcface_loss}
        optimizer = 'rmsprop'
        vector_size = vec_size #256 * 2

        # Fill return values
        for checkpoint in checkpoints:
            logger.info('Running checkpoint: ' + checkpoint)
            # Load and compile the trained network
            network_file = get_experiment_nets(checkpoint)
            model_full = load_model(network_file, custom_objects=custom_objects)
            model_full.compile(loss=self.loss, optimizer=optimizer, metrics=metrics)

            # Get a Model with the embedding layer as output and predict
            model_partial = Model(inputs=model_full.input, outputs=model_full.layers[out_layer].output)

            x_cluster_list = []
            y_cluster_list = []
            for x, y, s in zip(x_list, y_list, s_list):
                x_cluster = np.asarray(model_partial.predict(x))
                x_cluster_list.append(x_cluster)
                y_cluster_list.append(y)

            embeddings, speakers, num_embeddings = generate_embeddings(x_cluster_list, y_cluster_list, vector_size)

            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)

            # Calculate the time per utterance
            time = TimeCalculator.calc_time_all_utterances(y_cluster_list, seg_size)
            set_of_total_times.append(time)

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers, set_of_total_times


def prepare_data(X,y, segment_size):
    x, speakers = generate_test_data(X, y, segment_size)

    # Reshape test data because it is an lstm
    return x.reshape(x.shape[0], x.shape[3], x.shape[2]), speakers
