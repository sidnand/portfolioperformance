import numpy as np

from .model import Model
from src.utils.quadprog import *

class MinVarianceShortSellCon(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._name = 'Min Var Con'
        self._description = 'Minimum Variance with Short Sell Constraints'

    def _run(self, sigmaMLE):
        m, n = sigmaMLE.shape
        ub = np.ones((1, n))
        aeq = np.ones((1, n))
        beq = [1]
        lb = np.zeros((1, n))
        f = np.zeros((n, 1))

        solver = quadprog(sigmaMLE, f, aeq, beq, lb, ub)
        solverArr = np.asarray(solver)

        return solverArr