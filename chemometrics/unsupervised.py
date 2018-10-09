""" Unsupervised models. """
import numpy as np
from scipy.sparse.linalg import svds

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD


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


class PCA(BaseEstimator, TransformerMixin):

    """ Principal Components Analysis (PCA). 
    
    PCA is an unsupervised method for reducing the dimension
    of a data-matrix into a full-rank set of principal components to 
    provide an approximation of the input data matrix.
    
    Parameters
    ----------
    n_components : int
        Number of principal components to fit.
    algorithm : {'randomized', 'qr', 'nipals', 'svd'}, default 'svd'
        Algorithm to use for fitting PCA.
        
    Attributes
    ----------
    loadings_ : array
        Variable loading matrix, shape [p x n_components].
    R2_ : array
        Component-wise summary of fit.
        
    References
    ----------
    Wold, Svante, Kim Esbensen, and Paul Geladi. “Principal Component 
    Analysis.” Chemometrics and Intelligent Laboratory Systems, 
    Proceedings of the Multivariate Statistical Workshop for Geologists 
    and Geochemists, 2, no. 1 (August 1, 1987): 37–52. 
    https://doi.org/10.1016/0169-7439(87)80084-9.
    
    Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp. “Finding 
    Structure with Randomness: Probabilistic Algorithms for Constructing
    Approximate Matrix Decompositions.” arXiv:0909.4061 [Math], 
    September 22, 2009. http://arxiv.org/abs/0909.4061.

    Sharma, Alok, Kuldip K. Paliwal, Seiya Imoto, and Satoru Miyano. 
    “Principal Component Analysis Using QR Decomposition.” International 
    Journal of Machine Learning and Cybernetics 4, no. 6 
    (December 1, 2013): 679–83. https://doi.org/10.1007/s13042-012-0131-7.
    """

    def __init__(self, n_components, algorithm='svd'):
        self.n_components = n_components
        self.algorithm = algorithm

    def fit(self, X, skip_r2=False, **kwargs):
        """ Fit PCA using selected algoritm.
        
        Parameters
        ----------
        X : array
            Input data matrix, shape [n x p].
        skip_r2 : bool
            If True, skip calculation of R2.
        """
        if self.algorithm == 'randomized':
            loadings, R2 = self._fit_randomized_pca(X, **kwargs)
        elif self.algorithm == 'qr':
            loadings = self._fit_qr_pca(X, **kwargs)
            R2 = None
        elif self.algorithm == 'nipals':
            loadings = self._fit_nipals_pca(X, **kwargs)
            R2 = None
        elif self.algorithm == 'svd':
            loadings, R2 = self._fit_svd_pca(X, **kwargs)
        else:
            raise ValueError('Invalid algorithm: {}'.format(self.algorithm))

        self.loadings_ = loadings

        if not skip_r2:
            if R2 is not None:
                self.R2_ = R2
            else:
                self.R2_ = self._calculate_summary_of_fit(X)

        return self

    def transform(self, X):
        """ Transform input data into scores using fitted loadings.
        
        Parameters
        ----------
        X : array
            Input data matrix, shape [n x p]

        Returns
        -------
        scores : array
            Scores corresponding to `X`, shape [n x n_components]
        """
        return X.dot(self.loadings_)

    def inverse_transform(self, scores, component=None):
        """ Transform scores back into approximated data-space.
        
        Parameters
        ----------
        scores : array
            Obsevervation scores, shape [n x n_components].
        component : int, optional
            If component-index (1-indexed) is provided, restrict 
            inverse transform to only the current component.

        Returns
        -------
        X_approximated : array
            Approximated reconstruction of data, shape [n x p].
        """
        if component is not None:
            c_i = component - 1
            scores = scores[:, c_i][..., np.newaxis]
            loadings = self.loadings_[:, c_i][..., np.newaxis]
        else:
            loadings = self.loadings_

        return scores.dot(loadings.T)

    def _calculate_summary_of_fit(self, X):
        X_ss = (X ** 2).sum()
        scores = self.transform(X)
        R2 = list()

        for component in range(1, self.n_components + 1):
            X_approximated = self.inverse_transform(scores, component)
            residual_ss = ((X - X_approximated) ** 2).sum()
            R2.append(1 - residual_ss / X_ss)

        return np.array(R2)

    def _fit_svd_pca(self, X, **kwargs):
        U, S, V = svds(X, self.n_components)
        return np.fliplr(V.T), np.flip((S ** 2) / (X ** 2).sum(), 0)

    def _fit_qr_pca(self, X, **kwargs):
        Q, R = np.linalg.qr(X.T)
        U, S, V = svds(R.T, self.n_components)
        V = V.T
        loadings = Q.dot(np.fliplr(V))
        return loadings

    def _fit_nipals_pca(self, X, max_iter=1000, tol=1e-10):
        loadings = list()
        for k in range(self.n_components):
            loading = X[0, :][..., np.newaxis]
            loading /= np.linalg.norm(loading)
            for i in range(max_iter):
                previous_loading = loading

                scores = X.dot(loading)
                loading = X.T.dot(scores)
                loading /= np.linalg.norm(loading)

                if np.linalg.norm(loading - previous_loading) < tol:
                    loadings.append(loading)
                    break

            if k < self.n_components - 1:
                X = X - scores.dot(loading.T)

        return np.column_stack(loadings)

    def _fit_randomized_pca(self, X, **kwargs):
        pca = TruncatedSVD(self.n_components, **kwargs).fit(X)
        return pca.components_.T, pca.explained_variance_ratio_



class BidrectionalOrthogonalPLS(BaseEstimator, TransformerMixin):
    """ Bidirectional Orthogonal PLS (O2-PLS).
    
    Bidirectional Orthogonal PLS (O2-PLS) is a symmetric bi-directional 
    extension of Orthogonal PLS. O2-PLS decomposes two data matrices, X1 and X2,
    into systematic variation joint between the two matrices and systematic
    variation unique to each matrix.
    
    Parameters
    ----------
    n_joint_components : int
        Number of components describing joint variation.
    n_unique_components : tuple[int, int], default (0, 0)
        Number of components unique to each block to keep.
        
    Attributes
    ----------
    weights_ : list[array, array]
        X1- and X2-joint weight matrices. Shapes: [p1, n_joint_components] 
        and [p2, n_joint_components].
    loadings_ : list[array, array]
        X1- and X2-joint loading matrices. Shapes: [p1, n_joint_components] 
        and [p2, n_joint_components].
    correlation_loadings_ : list[array, array]
        X1- and X2-joint correlation loading matrices. Shapes: 
        [p1, n_joint_components] and [p2, n_joint_components].
    unique_weights_ : list[array, array]
        X1- and X2-unique weight matrices. Shapes: [p1, n_unique_componets[0]] 
        and [p2, n_unique_components[1]].
    unique_loadings_ : list[array, array]
        X1- and X2-unique loading matrices. Shapes: [p1, n_unique_componets[0]] 
        and [p2, n_unique_components[1]].
    regression_coefficients_ : list[array, array]
        X1- and X2-joint PLS regression coefficients. 
        Shapes [n_joint_components, n_joint_components]
    R2_ : list[array, array]
        Component-wise total R2 for each block.
    R2_joint_ : list[array, array]
        Component-wise joint R2 for each block.
    R2_unique : list[array, array]
        Component-wise unique R2 for each block.
        
    References
    ----------
    Trygg, Johan. “O2-PLS for Qualitative and Quantitative Analysis in 
    Multivariate Calibration.” Journal of Chemometrics 16, no. 6 
    (June 1, 2002): 283–93. https://doi.org/10.1002/cem.724.
    
    Trygg, Johan, and Svante Wold. “O2-PLS, a Two-Block (X–Y) Latent 
    Variable Regression (LVR) Method with an Integral OSC Filter.” Journal 
    of Chemometrics 17, no. 1 (January 1, 2003): 53–64. 
    https://doi.org/10.1002/cem.775.
    """

    def __init__(self,
                 n_joint_components,
                 n_unique_components=(0, 0)):
        self.n_joint_components = n_joint_components
        self.n_unique = n_unique_components

    def fit(self, X1, X2):
        """ Fit model joint data matrices `X1` and `X2`
         
        Parameters
        ----------
        X1, X2 : array
            Data matrices, shapes [n, p1] and [n, p2]
        """
        # Do singular value decomposition.
        W2, S, _W1 = svds(X2.T.dot(X1), k=self.n_joint_components)

        sort_i = np.argsort(S)[::-1]
        W1 = _W1.T[:, sort_i]
        W2 = W2[:, sort_i]

        # Calculate initial scores.
        T1_initial = X1.dot(W1)
        T2_initial = X2.dot(W2)

        # Remove orthogonal variation from data blocks.
        results_1 = self._remove_orthogonal_variation(X1, T1_initial, W1, self.n_unique[0])
        T1, P1, P1_corr, P1_ortho, W1_ortho, R2j_1, R2u_1, R2_1 = results_1

        results_2 = self._remove_orthogonal_variation(X2, T2_initial, W2, self.n_unique[1])
        T2, P2, P2_corr, P2_ortho, W2_ortho, R2j_2, R2u_2, R2_2 = results_2

        B1 = np.linalg.inv(T1.T.dot(T1)).dot(T1.T).dot(T2)
        B2 = np.linalg.inv(T2.T.dot(T2)).dot(T2.T).dot(T1)

        self.weights_ = [W1, W2]
        self.loadings_ = [P1, P2]
        self.correlation_loadings_ = [P1_corr, P2_corr]
        self.unique_weights_ = [W1_ortho, W2_ortho]
        self.unique_loadings_ = [P1_ortho, P2_ortho]
        self.regression_coefficients_ = [B1, B2]
        self.R2_ = [np.array(R2_1), np.array(R2_2)]
        self.R2_joint_ = [np.array(R2j_1), np.array(R2j_2)]
        self.R2_unique_ = [np.array(R2u_1), np.array(R2u_2)]

        return self

    def transform(self, X1=None, X2=None, return_unique=True):
        """ Use fitted parameters to transform data matrix onto model space.
        
        Parameters
        ----------
        X1, X2 : array, optional
            Data matrices. Number of columns must match column dimensions
            of data used for fitting model.
        return_unique : bool, default True
            If True, return tuple of joint scores and unique scores. 
            Otherwise return only list of joint scores for each data matrix
            
        Returns
        -------
        scores : list[array | None] or tuple[list[array | None]]
            Joint scores, or joint and unique scores for each block. 
            None is returned for each data matrix not provided.
        """
        transformation = []
        unique_transformation = []

        for X, W_u, P_u, W, B, W_other in zip([X1, X2],
                                               self.unique_weights_,
                                               self.unique_loadings_,
                                               self.weights_,
                                               self.regression_coefficients_,
                                               self.weights_[::-1]):

            if X is None:
                transformation.append(None)
                unique_transformation.append(None)
                continue

            # If one-dimensional, add dimension to make into row-vector.
            if X.ndim == 1:
                X = X[np.newaxis, :]

            # Remove orthogonal variation based on model-weights and loadings.
            T_ortho = np.zeros((X.shape[0], W_u.shape[1]))
            for i, (wu, pu) in enumerate(zip(W_u.T, P_u.T)):

                t_op = (X.dot(wu)) * (1 / wu.dot(wu))
                T_ortho[:, i] = t_op

                X = X - np.outer(t_op, pu)

            # Predict scores from data with orthogonal variation removed.
            T_pred = X.dot(W).dot(np.linalg.inv(W.T.dot(W)))
            transformation.append(T_pred)
            unique_transformation.append(T_ortho)

        if return_unique:
            return transformation, unique_transformation
        else:
            return transformation

    def predict(self, X1=None, X2=None):
        """ Predict the opposite block from input data matrix.
         
        Parameters
        ----------
        X1, X2 : array, optional
            Data matrices. Number of columns must match column dimensions
            of data used for fitting model.
        
        Returns
        -------
        predictions : list[array | None]
            Prediction of the opposite for each input matrix. None for
            data matrices not provided.
        """
        T_pred = self.transform(X1, X2, return_unique=False)

        predictions = []
        for T, B, W_other in zip(T_pred,
                                 self.regression_coefficients_,
                                 self.weights_[::-1]):

            if T is not None:
                predicted = T.dot(B).dot(W_other.T)
                predictions.append(predicted)
            else:
                predictions.append(None)

        return predictions

    def _remove_orthogonal_variation(self, X, T, W, n_unique):
        """ Remove the orthogonal variation from data matrix X and adjust
        scores and loadings.
        
        Parameters
        ----------
        X : array
            Data matrix, shape [n, p]
        T :  array
            Initial joint scores, shape [n, n_joint_components]
        W : array
            Joint PLS weights, shape [p, n_joint_components]
        n_unique : int
            Number of unique components to use in block.
        """
        x_ss = (X ** 2).sum()  # Sum of squares.
        E = X - T.dot(W.T)  # Residuals.

        # Initialize dictionary containing explained variation.
        r2x = np.sum((T.dot(W.T) ** 2)) / x_ss

        R2_joint = [r2x]
        R2_unique = [0]
        R2 = [r2x]  # Total variance explained.

        # Initialize ortho-component matrices.
        T_unique = np.zeros((X.shape[0], n_unique))
        P_unique = np.zeros((X.shape[1], n_unique))
        W_unique = np.zeros((X.shape[1], n_unique))

        # Remove Y-orthogonal variation.
        for i in range(n_unique):

            w_o = np.squeeze(PCA(1, 'nipals').fit_transform(E.T.dot(T)))
            w_o = w_o / np.linalg.norm(w_o)

            t_o = X.dot(w_o)
            p_o = X.T.dot(t_o / (t_o.T.dot(t_o)))

            # Deflate X and update joint scores.
            X = X - np.outer(t_o, p_o)
            T = X.dot(W)

            R2_joint.append((T.dot(W.T) ** 2).sum() / x_ss)

            # Update matrices.
            T_unique[:, i] = t_o
            P_unique[:, i] = p_o
            W_unique[:, i] = w_o

            # Deflate.
            X_unique = T_unique.dot(P_unique.T)
            E = X - T.dot(W.T) - X_unique

            # Update explained variation.
            R2.append(1 - (E ** 2).sum() / x_ss)
            R2_unique.append((X_unique ** 2).sum() / x_ss)

        # Calculate correlation loadings.
        T_norm_sq = np.apply_along_axis(lambda a: a / a.T.dot(a), 0, T)
        P = X.T.dot(T_norm_sq)
        P_corr = np.zeros(W.shape)

        for t in T.T:
            p_c = np.apply_along_axis(lambda v: _cross_div_norms(v, t), 0, X)
            P_corr[:, i] = p_c

        return T, P, P_corr, P_unique, W_unique, R2_joint, R2_unique, R2


def _cross_div_norms(vec1, vec2):
    """ Divide scalar product of vectors with product of norms."""
    cross = vec2.T.dot(vec1)
    norm1 = np.sqrt(vec1.T.dot(vec1))
    norm2 = np.sqrt(vec2.T.dot(vec2))

    return cross / (norm1 * norm2)
