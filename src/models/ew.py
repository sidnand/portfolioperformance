import numpy as np

from .model import Model

class EqualWeightModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'Equal Weight'
        self._description = 'Equal Weight Model'

    def _run(self, cols):
        return 1/cols * np.ones((cols - 1, 1))