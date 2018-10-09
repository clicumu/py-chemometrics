""" Regression models. """

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,\
    ClassifierMixin


class PLS(BaseEstimator, TransformerMixin, RegressorMixin):

    def __init__(self, n_components, algorithm='nipals'):
        super(PLS, self).__init__()
        self.n_components = n_components
        self.algorithm = algorithm

    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = Y[..., np.newaxis]

        if self.algorithm == 'nipals':
            weights, x_loadings, y_weights = self._fit_nipals_pls(X, Y)
        else:
            raise ValueError('Invalid algorithm: {}'.format(self.algorithm))

        W_star = weights.dot(np.linalg.pinv(x_loadings.T.dot(weights)))
        coefficients = W_star.dot(y_weights.T)

        self.weights_ = weights
        self.loadings_ = x_loadings
        self.y_weights_ = y_weights
        self.coefficients_ = coefficients

        return self

    def _fit_nipals_pls(self, X, Y, n_iter=1000, tol=1e-8):
        weights = np.zeros((X.shape[1], self.n_components))
        x_loadings = np.zeros((X.shape[1], self.n_components))
        y_weights = np.zeros((Y.shape[1], self.n_components))

        for i in range(self.n_components):
            y_score = Y[:, 0][..., np.newaxis]
            for _ in range(n_iter):
                x_weight = X.T.dot(y_score) / y_score.T.dot(y_score)
                x_weight /= np.linalg.norm(x_weight)
                x_score = X.dot(x_weight) / x_weight.T.dot(x_weight)

                y_weight = Y.T.dot(x_score) / x_score.T.dot(x_score)
                y_weight /= np.linalg.norm(y_weight)

                new_y_score = Y.dot(y_weight) / y_weight.T.dot(y_weight)

                if np.linalg.norm(new_y_score - y_score) < tol:
                    x_loading = X.T.dot(x_score) / x_score.T.dot(x_score)
                    x_loading_norm = np.linalg.norm(x_loading)
                    x_loading /= x_loading_norm

                    X = X - x_score.dot(x_loading.T)
                    Y = Y - x_score.dot(y_weight.T)

                    weights[:, i] = x_weight.squeeze() * x_loading_norm
                    x_loadings[:, i] = x_loading.squeeze()
                    y_weights[:, i] = y_weight.squeeze()
                    break
                else:
                    y_score = new_y_score

        return weights, x_loadings, y_weights


class PLS_DA(ClassifierMixin, PLS):

    def fit(self, X, Y):
        NotImplemented


class OrthogonalPLS(PLS):

    """
    Trygg, Johan, and Svante Wold. “Orthogonal projections to latent structures
    (O-PLS).” Journal of Chemometrics 16, no. 3 (March 1, 2002): 119–28.
    https://doi.org/10.1002/cem.695.
    """

    def __init__(self, n_components, n_orthogonal_components=0, **kwargs):
        super(OrthogonalPLS, self).__init__(n_components, **kwargs)
        self.n_orthogonal_components = n_orthogonal_components

    def fit(self, X, Y):
        if self.n_orthogonal_components == 0:
            super(OrthogonalPLS, self).fit(X, Y)
            self.orthogonal_loadings_ = None
            self.orthogonal_weights_ = None
        else:
            if Y.ndim == 1 or Y.shape[1] == 1:
                fitting_results = self._fit_single_y(X, Y.squeeze()[..., np.newaxis])
            else:
                raise NotImplementedError('Multi-response Y not implemented.')

            weights, loadings, y_weights, o_weights, o_loadings = fitting_results
            self.weights_ = weights
            self.loadings_ = loadings
            self.y_weights_ = y_weights
            self.orthogonal_weights_ = o_weights
            self.orthogonal_loadings_ = o_loadings

        W_star = self.weights_.dot(np.linalg.pinv(self.loadings_.T.dot(self.weights_)))
        coefficients = W_star.dot(self.y_weights_.T)
        self.coefficients_ = coefficients

        return self

    def transform(self, X):
        X_orthogonal_removed, ortho_scores = self.remove_orthogonal_variation(X)
        scores = X_orthogonal_removed.dot(self.weights_)
        return np.vstack([scores, ortho_scores])

    def remove_orthogonal_variation(self, X):
        ortho_scores = X.dot(self.orthogonal_weights_)
        X_orthogonal_removed = X - ortho_scores.dot(self.orthogonal_loadings_.T)
        return X_orthogonal_removed, ortho_scores

    def predict(self, X):
        X_orthogonal_removed, ortho_scores = self.remove_orthogonal_variation(X)
        return X_orthogonal_removed.dot(self.coefficients_)

    def _fit_single_y(self, X, y):
        assert y.shape == (X.shape[0], 1)
        n_ortho = self.n_orthogonal_components

        assert n_ortho >= 1
        orthogonal_weights = np.zeros((X.shape[1], n_ortho))
        orthogonal_loadings = np.zeros((X.shape[1], n_ortho))

        # Following Trygg & Wold 2002.
        for i_ortho in range(n_ortho):
            w, p, c = self._single_nipals_step(X, y)

            # OSC
            w_ortho = p - (w.T.dot(p) / (w.T.dot(w))) * (w)
            w_ortho /= np.linalg.norm(w_ortho)
            orthogonal_weights[:, i_ortho] = w_ortho.squeeze()

            t_ortho = X.dot(w_ortho)

            np.testing.assert_almost_equal(t_ortho.T.dot(y), np.array([[0]]))
            p_ortho = ((t_ortho.T.dot(X)) / t_ortho.T.dot(t_ortho)).T
            orthogonal_loadings[:, i_ortho] = p_ortho.squeeze()

            X = X - t_ortho.dot(p_ortho.T)

        w, p, c = self._single_nipals_step(X, y)

        return w, p, c, orthogonal_weights, orthogonal_loadings

    def _single_nipals_step(self, X, y):
        w = (y.T.dot(X) / y.T.dot(y)).T
        w /= np.linalg.norm(w)
        t = X.dot(w) / (w.T.dot(w))
        c = (t.T.dot(y) / (t.T.dot(t))).T
        u = y.dot(c) / (c.T.dot(c))
        p = (t.T.dot(X) / t.T.dot(t)).T
        return w, p, c


class OrthogonalPLS_DA(ClassifierMixin, OrthogonalPLS):

    def fit(self, X, Y):
        NotImplemented


if __name__ == '__main__':
    t = np.random.randn(20, 1) * 5
    u = t #+ np.random.randn(20, 1)
    p = np.random.randn(40, 1)
    c = np.random.randn(5, 1)

    to = np.sin(np.linspace(0, 6, 20))[..., np.newaxis]
    po = np.random.randn(40, 1)

    X = t.dot(p.T) + to.dot(po.T)
    Y = u.dot(c.T)

    pls = OrthogonalPLS(1, 1).fit(X, u)