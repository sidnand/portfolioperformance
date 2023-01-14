import numpy as np

from ...modelGamma import ModelGamma

class BayesStein(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, currentGamma, invSigmaBS, muBS):
        # muBS = np.expand_dims(muBS, axis=1)
        return (1/currentGamma) * invSigmaBS @ muBS[1:]