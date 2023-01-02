import numpy as np

from ..model import Model
from ..utils.quadprog import quadprog

class MinVarShortSellCon(Model):
    def __init__(self, name):
        super().__init__(name)

    def _run(self, sigmaMLE, currentSubset, period, nSubsets):
        alpha = self._alpha(sigmaMLE)

        self.weights[:, currentSubset] = alpha[:, 0]

        if currentSubset == 0:
            self.weightsBuyHold[:, currentSubset] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset] = self.buyHold(
                self.weights[:, currentSubset - 1], currentSubset, period)

        if (nSubsets > 1):
            self.outSample[:, currentSubset] = self.outOfSampleReturns(alpha, currentSubset, period)[:, 0]

    def _alpha(self, sigmaMLE):
        period, n = sigmaMLE.shape
        ub = np.ones((1, n))
        aeq = np.ones((1, n))
        beq = [1]
        lb = np.zeros((1, n))
        f = np.zeros((n, 1))

        solver = quadprog(sigmaMLE, f, aeq, beq, lb, ub)
        solverArr = np.asarray(solver)

        return solverArr