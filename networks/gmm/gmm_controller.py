
from common.network_controller import NetworkController
from sklearn import mixture
from common.utils.pickler import load, save
from common.utils.paths import get_experiment_nets, get_speaker_pickle
from common.clustering.generate_embeddings import generate_embeddings
from common.utils.ShortUtteranceConverter import create_data_lists
import numpy as np


class GMMController(NetworkController):
    def __init__(self, config):
        super().__init__("gmm", config)
        self.network_file = self.name + "_100"

    def train_network(self):
        mixture_count = self.config.getint('gmm', 'mixturecount')
        X, y, speaker_names = load(get_speaker_pickle(self.config.get('train', 'pickle')+'_mfcc'))
        model = []

        for i in range(len(X)):
            features = X[i]
            gmm = mixture.GaussianMixture(n_components=mixture_count, covariance_type='diag', n_init=1)
            gmm.fit(features.transpose())
            speaker = {'mfccs': features, 'gmm': gmm}
            model.append(speaker)

        save(model, get_experiment_nets(self.name))

    def get_embeddings(self):
        X_train, y_train, speaker_train_names = load(get_speaker_pickle(self.get_validation_data_name()+'_train_mfcc'))
        X_test, y_test, speaker_test_names = load(get_speaker_pickle(self.get_validation_data_name()+'_test_mfcc'))

        model = load(get_experiment_nets(self.name))

        set_of_embeddings = []
        set_of_speakers = []
        set_of_num_embeddings = []

        train_outputs = self.generate_outputs(X_train, model)
        test_outputs = self.generate_outputs(X_test, model)

        set_of_times = [np.zeros((len(y_test) + len(y_train)), dtype=int)]

        outputs, y_list, s_list = create_data_lists(False, train_outputs,test_outputs,y_train,y_test)

        embeddings, speakers, number_embeddings = generate_embeddings(outputs, y_list, len(model))

        set_of_embeddings.append(embeddings)
        set_of_speakers.append(speakers)
        set_of_num_embeddings.append(number_embeddings)
        checkpoints = [self.network_file]

        return checkpoints, set_of_embeddings, set_of_speakers, set_of_num_embeddings, set_of_times

    def generate_outputs(self, X, model):
        embeddings = []
        for sample in X:
            reshape_sample = sample.transpose()
            score_vector = []

            for speaker in model:  # find the most similar known speaker for the given test sample of a voice
                score_per_featurevector = speaker['gmm'].score(reshape_sample)  # yields log-likelihoods per feature vector
                score_vector.append(score_per_featurevector)

            embeddings.append(score_vector)

        return embeddings
