import numpy as np
from sklearn.neural_network import MLPClassifier


class NeuralNet:
    """
    An attempt to interface MLPClassifier in a way that makes the hyperparameters related to width/depth more clear.
    We intercept the parameters fetched from MLPClassifier, and translates the parameters into our own
    Before going back to the same as before
    """
    def __init__(self, **params):
        self.neuralnet = MLPClassifier(**params)

    def fit(self, X, y, **params):
        fit = self.neuralnet.fit(X, y)
        return fit

    def predict(self, X):
        return self.neuralnet.predict(X)

    def predict_log_proba(self, X):
        return self.neuralnet.predict_log_proba(X)

    def predict_proba(self, X):
        return self.neuralnet.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        return self.neuralnet.score(X, y, sample_weight)

    def set_params(self, **params):
        n_hidden_neurons = params['n_hidden_neurons']
        n_hidden_layers = params['n_hidden_layers']
        hidden_layer_sizes = tuple([n_hidden_neurons for i in range(n_hidden_layers)])

        params['hidden_layer_sizes'] = hidden_layer_sizes
        del params['n_hidden_neurons']
        del params['n_hidden_layers']

        self.neuralnet.set_params(**params)

    def get_params(self, deep=True):
        params = self.neuralnet.get_params(deep)
        del params['hidden_layer_sizes']
        return params

