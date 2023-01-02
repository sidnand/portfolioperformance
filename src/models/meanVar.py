import numpy as np

from ..model import Model

class MeanVar(Model):
    def __init__(self, name):
        super().__init__(name)

    def _run(self, gamma, invSigmaMLE, mu, currentSubset, period):
        for i in gamma:
            #  : Mean Variance
            alpha = self.alpha(i, invSigmaMLE, mu)
            self.weights[:, currentSubset] = alpha[:, 0]
            self.outSample[:, currentSubset] = self.outOfSampleReturns(
                alpha, currentSubset, period)[:, 0]

        if currentSubset == 0:
            self.weightsBuyHold[:, currentSubset] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset] = self.buyHold(
                self.weights[:, currentSubset - 1], currentSubset, period)

    def alpha(self, gamma, invSigmaMLE, mu):
        alpha = (1/gamma) * invSigmaMLE @ mu[1:]
        return alpha.reshape(alpha.shape[0],-1)