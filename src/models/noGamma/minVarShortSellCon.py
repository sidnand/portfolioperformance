import numpy as np

from ...modelNoGamma import ModelNoGamma
from ...utils.quadprog import quadprog
from ...utils.sharedOptions import minVarConOptions

class MinVarShortSellCon(ModelNoGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, sigmaMLE):
        n, ub, aeq, beq, f = minVarConOptions(sigmaMLE)
        lb = np.zeros((1, n))

        solver = quadprog(sigmaMLE, f, aeq, beq, lb, ub)
        solverArr = np.asarray(solver)

        return solverArr