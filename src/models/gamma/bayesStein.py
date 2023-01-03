import numpy as np

from ...model import *

class BayesStein(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, currentGamma, invSigmaBS, muBS):
        return (1/currentGamma) * invSigmaBS @ muBS[1:]