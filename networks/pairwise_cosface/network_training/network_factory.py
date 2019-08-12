"""
    This file can create various networks. They are forwarded to
    the clustering_network as a network function.

    Work of Lukic and Vogt.
"""
import lasagne
import theano.tensor as T
import numpy as np

from common.utils.load_config import *
from common.utils.paths import *
from lasagne.layers import Layer

config = load_config(None, join(get_common(), 'config.cfg'))
seg_size = config.getint('pairwise_cosface', 'seg_size')
spectrogram_height = config.getint('pairwise_cosface', 'spectrogram_height')


class CosFaceDense(Layer):
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 num_leading_axes=1, **kwargs):
        super(CosFaceDense, self).__init__(incoming, **kwargs)

        self.num_units = num_units
        self.num_leading_axes = num_leading_axes
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)
        x = lasagne.regularization.l2(input)
        W = lasagne.regularization.l2(self.W)
        return T.dot(x, W)


def create_network_10_speakers(input_var):
    return create_network_n_speakers(input_var=input_var, n=10)


def create_network_100_speakers(input_var):
    return create_network_n_speakers(input_var=input_var, n=100)

def create_network_470_speakers(input_var):
    return create_network_n_speakers(input_var=input_var, n=470)


def create_network_590_speakers(input_var):
    return create_network_n_speakers(input_var=input_var, n=590)


def create_network_n_speakers(input_var, n):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, spectrogram_height, seg_size), input_var=input_var)

    # convolution layers 1
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(4, 4),
                                         nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 4), stride=(2, 2))

    # convolution layers 2
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(4, 4),
                                         nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 4), stride=(2, 2))

    # dense layer
    network = lasagne.layers.DenseLayer(network, num_units=n * 10, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(network, num_units=n * 5, nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    network = CosFaceDense(network, num_units=n)

    return network


def create_network_KL_clustering_no_convolution(input_var, input_size=1000, output_size=100):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 1, input_size),
                                        input_var=input_var)

    # dense layer
    network = lasagne.layers.DenseLayer(network, num_units=input_size * 10, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(network, num_units=input_size * 5, nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    network = lasagne.layers.DenseLayer(network, num_units=output_size, nonlinearity=lasagne.nonlinearities.softmax)

    return network


def create_network_clustering_segment_embedding(input_var, input_size=100, max_segments=1000):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, max_segments, input_size),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3, input_size),
                                         nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # dense layer
    network = lasagne.layers.DenseLayer(network, num_units=input_size * 10, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.BatchNormLayer(network)
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(network, num_units=input_size * 5, nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    network = lasagne.layers.DenseLayer(network, num_units=input_size, nonlinearity=lasagne.nonlinearities.softmax)

    return network


def create_identification_network(num_speakers):
    network = lasagne.layers.InputLayer(shape=(None, 1, spectrogram_height, seg_size))

    # convolution layers 1
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(4, 4))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 4), stride=(2, 2))

    # convolution layers 2
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(4, 4))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(4, 4), stride=(2, 2))

    # dense layer
    network = lasagne.layers.DenseLayer(network, num_units=num_speakers * 10)
    network = lasagne.layers.DropoutLayer(network, p=0.5)
    network = lasagne.layers.DenseLayer(network, num_units=num_speakers * 5)

    # output layer
    network = lasagne.layers.DenseLayer(network, num_units=num_speakers, nonlinearity=lasagne.nonlinearities.softmax)
