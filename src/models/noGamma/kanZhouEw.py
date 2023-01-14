import numpy as np

from ...modelNoGamma import ModelNoGamma


class KanZhouEw(ModelNoGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, nRisky, nPoints, sigma):
        invSigma = np.linalg.inv(sigma)

        esige_matrix = (np.ones((1, nRisky)) @ sigma @ np.ones((nRisky, 1)))
        einvsige_matrix = (np.ones((1, nRisky)) @ invSigma @ np.ones((nRisky, 1)))

        esige = esige_matrix[0][0]
        einvsige = einvsige_matrix[0][0]

        k = (nPoints**2 * (nPoints-2)) / ((nPoints-nRisky-1) * (nPoints-nRisky-2) * (nPoints-nRisky-4))

        d = ((nPoints - nRisky - 2) * esige * einvsige - nRisky**2 * nPoints)/(nRisky**2 * (nPoints - nRisky - 2) * k * einvsige - 2 * nPoints * nRisky**2 * einvsige + (nPoints - nRisky - 2) * einvsige**2 * esige)
        c = 1 - d * einvsige

        alpha = (c * (1 / nRisky) * np.ones((nRisky,1))) + ((d * invSigma) @ np.ones((nRisky,1)))

        return alpha