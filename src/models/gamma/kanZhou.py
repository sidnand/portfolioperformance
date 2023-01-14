import numpy as np
import scipy as sp

from ...modelGamma import ModelGamma


class KanZhou(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, currentGamma, invSigmaMLE, mu, period, nRisky):
        # mu = np.expand_dims(mu[1:], axis=1)
        mu = mu[1:]

        mug = (mu.T @ invSigmaMLE @ np.ones((nRisky, 1))) / np.ones((1, nRisky)) @ invSigmaMLE @ np.ones((nRisky, 1))
        phiHat2 = (mu - mug).T @ invSigmaMLE @ (mu - mug)
        x = phiHat2 / (1 + phiHat2)

        B = sp.special.betainc((nRisky - 1) / 2, (period - nRisky + 1) / 2, x)
        phiHat2a = ((period - nRisky - 1) * phiHat2 - (nRisky - 1)) / period + (2 * (phiHat2) ** ((nRisky - 1) / 2) * (1 + phiHat2) ** (-(period - 2) / 2)) / (period * B)
        c3 = ((period - nRisky - 1) * (period - nRisky - 4)) / (period * (period - 2))
        denom = phiHat2a + nRisky / period

        alpha = c3 / currentGamma * ((phiHat2a / denom) * invSigmaMLE @ mu + ((nRisky / period) / denom) * mug * invSigmaMLE @ np.ones((nRisky, 1)))

        return alpha
