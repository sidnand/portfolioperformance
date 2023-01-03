from ..model import *


class MeanVar(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, currentGamma, invSigmaMLE, mu):
        alpha = (1/currentGamma) * invSigmaMLE @ mu[1:]
        return alpha.reshape(alpha.shape[0],-1)