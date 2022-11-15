import numpy as np

from .model import Model

class EqualWeightModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'EqW'
        self._description = '1 / N Equal Weight Portfolio'

    def _run(self, cols):
        return 1/cols * np.ones((cols - 1, 1))