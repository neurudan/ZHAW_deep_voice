
from common.network_controller import NetworkController
from sklearn import mixture
from common.utils.pickler import load, save
from common.utils.paths import get_experiment_nets, get_speaker_pickle
from common.clustering.generate_embeddings import generate_embeddings


class GMMController(NetworkController):
    def __init__(self, config):
        super().__init__("gmm", config)
        self.network_file = self.name + "_100"

    def train_network(self):
        '''build a 16 component diagonal covariance GMM from the given features (usually 13 MFCCs)'''
        mixture_count = self.config.getint('gmm', 'mixturecount')
        X, y, speaker_names = load(get_speaker_pickle(self.config.get('train', 'mfcc_pickle')))
        model = []

        for i in range(len(X)):
            features = X[i]
            gmm = mixture.GaussianMixture(n_components=mixture_count, covariance_type='diag', n_init=1)
            gmm.fit(features.transpose())
            speaker = {'mfccs': features, 'gmm': gmm}
            model.append(speaker)

        save(model, get_experiment_nets(self.name))

    def get_embeddings(self):
        X_train, y_train, speaker_train_names = load(self.get_validation_train_data())
        X_test, y_test, speaker_test_names = load(self.get_validation_test_data())

        model = load(get_experiment_nets(self.name))

        train_outputs = self.generate_outputs(X_train, model)
        test_outputs = self.generate_outputs(X_test, model)

        generate_embeddings(train_outputs, test_outputs, y_train, y_test, len(model))

    def generate_outputs(self, X, model):
        embeddings = []
        for sample in X:
            sample = sample.reshape(sample.shape[1:]).transpose()
            score_vector = []

            for speaker in model:  # find the most similar known speaker for the given test sample of a voice
                score_per_featurevector = speaker['gmm'].score(sample)  # yields log-likelihoods per feature vector
                score_vector.append(score_per_featurevector)

            embeddings.append(score_vector)

        return embeddings