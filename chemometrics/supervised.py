""" Regression models. """

from sklearn.base import ClassifierMixin
from sklearn.cross_decomposition import PLSRegression


PLS = PLSRegression


class PLS_DA(ClassifierMixin, PLSRegression):

    def fit(self, X, Y):
        NotImplemented


class OrthogonalPLS(PLSRegression):

    def __init__(self, *args, n_orthogonal_components=0, **kwargs):
        super(OrthogonalPLS, self).__init__(*args, **kwargs)
        self.n_orthogonal_components = n_orthogonal_components

    def fit(self, X, Y):
        super(OrthogonalPLS, self).fit(X, Y)  # Fit PLS.

        # PLS is fitted. Implement calculation of orthogonal components.

        return self


class OrthogonalPLS_DA(ClassifierMixin, OrthogonalPLS):

    def fit(self, X, Y):
        NotImplemented