import numpy as np

from .model import Model
from src.utils.quadprog import *

class JagannathanMa(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'Jagannathan Ma'
        self._description = 'Jagannathan Ma Portfolio Model'

    def _run(self, sigma):
        m, n = sigma.shape
        ub = np.ones((1, n))
        aeq = np.ones((1, n))
        beq = [1]
        lb = np.ones((1, n)) / (2 * n)
        f = np.zeros((n, 1))

        solver = quadprog(sigma, f, aeq, beq, lb, ub)
        solverArr = np.asarray(solver)

        return solverArr