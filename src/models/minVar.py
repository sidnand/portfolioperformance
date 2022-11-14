import numpy as np

from .model import Model

class MinVariance(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'Min Variance'

    def _run(self, invSigmaMLE, AMLE, cols):
        return 1/AMLE * np.dot(invSigmaMLE, np.ones((cols - 1, 1)))