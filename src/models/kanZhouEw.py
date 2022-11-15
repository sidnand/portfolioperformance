import numpy as np

from .model import Model
from src.utils.quadprog import *

class KanZhouEqualWeight(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'KanZhouEqW'
        self._description = 'Kan Zhou Equal Weight Portfolio'

    def _run(self, sigma, N, M):
        invSigmaMLE = np.linalg.inv(sigma)

        esige = (np.ones((1, N)) @ sigma @ np.ones((N, 1)))[0][0]
        einvsige = (np.ones((1, N)) @ invSigmaMLE @ np.ones((N, 1)))[0][0]

        k = (M^2 * (M-2)) / ((M-N-1) * (M-N-2) * (M-N-4))

        d = ((M - N - 2) * esige * einvsige - N**2 * M)/(N**2 * (M - N - 2) * k * einvsige - 2 * M * N**2 * einvsige + (M - N - 2) * einvsige**2 * esige)
        c = 1 - d * einvsige

        # alpha = c * 1 / N * np.ones((N,1)) + d * invSigmaMLE @ np.ones((N,1))

        alpha = (c * (1 / N) * np.ones((N,1))) + ((d * invSigmaMLE) @ np.ones((N,1)))

        return alpha