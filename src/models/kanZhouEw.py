import numpy as np

from ..model import Model

class KanZhouEw(Model):
    def __init__(self, name):
        super().__init__(name)

    def _run(self, nRisky, period, sigma, currentSubset, nSubsets):
        alpha = self._alpha(nRisky, period, sigma)

        self.weights[:, currentSubset] = alpha[:, 0]

        if currentSubset == 0:
            self.weightsBuyHold[:, currentSubset] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset] = self.buyHold(
                self.weights[:, currentSubset - 1], currentSubset, period)

        if (nSubsets > 1):
            self.outSample[:, currentSubset] = self.outOfSampleReturns(alpha, currentSubset, period)[:, 0]

    def _alpha(self, nRisky, period, sigma):
        invSigma = np.linalg.inv(sigma)

        esige_matrix = (np.ones((1, nRisky)) @ sigma @ np.ones((nRisky, 1)))
        einvsige_matrix = (np.ones((1, nRisky)) @ invSigma @ np.ones((nRisky, 1)))

        esige = esige_matrix[0][0]
        einvsige = einvsige_matrix[0][0]

        k = (period**2 * (period-2)) / ((period-nRisky-1) * (period-nRisky-2) * (period-nRisky-4))

        d = ((period - nRisky - 2) * esige * einvsige - nRisky**2 * period)/(nRisky**2 * (period - nRisky - 2) * k * einvsige - 2 * period * nRisky**2 * einvsige + (period - nRisky - 2) * einvsige**2 * esige)
        c = 1 - d * einvsige

        alpha = (c * (1 / nRisky) * np.ones((nRisky,1))) + ((d * invSigma) @ np.ones((nRisky,1)))

        return alpha