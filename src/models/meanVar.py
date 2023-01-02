import numpy as np

from ..model import *


class MeanVar(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, gamma, invSigmaMLE, mu):
        alpha = (1/gamma) * invSigmaMLE @ mu[1:]
        return alpha.reshape(alpha.shape[0],-1)