""" Unsupervised models. """
import numpy as np
from scipy.sparse.linalg import svds

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

            w_o = np.squeeze(PCA(n_components=1).fit_transform(E.T.dot(T)))
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
