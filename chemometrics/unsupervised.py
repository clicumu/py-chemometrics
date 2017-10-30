""" Unsupervised models. """

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class MCR(BaseEstimator, TransformerMixin):

    """ Multivariate Curve Resolution. """

    def __init__(self, n_components=None):
        super(MCR, self).__init__()
        self.n_components = n_components

    def fit(self):
        """ Fit MCR model to data. """

    def score(self, X, y=None):
        """ Score samples. """

    def transform(self, X):
        """ Transform X. """


class NipalsPCA(PCA):

    """ PCA using NIPALS algorithm. """

    def _fit(self, X):
        """ Fit PCA using NIPALS algorithm. """
