import numpy as np

from ..modelNoGamma import ModelNoGamma
from ..utils.statistics import *

class EqualWeight(ModelNoGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, n):
        return (1/n) * np.ones((n - 1, 1))