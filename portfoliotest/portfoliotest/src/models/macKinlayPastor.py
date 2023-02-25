import numpy as np

from ..modelGamma import ModelGamma


class MacKinlayPastor(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, currentGamma, sigmaMLE, mu):
        mu = mu[1:]

        uhat = sigmaMLE + np.outer(mu, mu)
        (D, V) = np.linalg.eig(uhat)
        d = np.diag(D)
        
        l1 = np.max(d)
        q1 = V[:, np.nonzero(d == l1)[0]]

        muTilde = (q1.T @ mu) * q1

        return (1 / currentGamma * muTilde) / (l1 - muTilde.T @ muTilde)
