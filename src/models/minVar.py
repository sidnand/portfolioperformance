import numpy as np

from ..model import Model

class MinVar(Model):
    def __init__(self, name):
        super().__init__(name)

    def _run(self, invSigmaMLE, amle, n, currentSubset, period, nSubsets):
        alpha = self._alpha(invSigmaMLE, amle, n)

        self.weights[:, currentSubset] = alpha[:, 0]

        if currentSubset == 0:
            self.weightsBuyHold[:, currentSubset] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset] = self.buyHold(self.weights[:, currentSubset - 1], currentSubset, period)

        if (nSubsets > 1):
            self.outSample[:, currentSubset] = self.outOfSampleReturns(alpha, currentSubset, period)[:, 0]

    def _alpha(self, invSigmaMLE, amle, n):
        return (1/amle) * invSigmaMLE @ np.ones((n - 1, 1))