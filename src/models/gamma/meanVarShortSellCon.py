import numpy as np

from ...modelGamma import ModelGamma
from ...utils.quadprog import *
from ...utils.sharedOptions import gammaShortSellConOptions


class MeanVarShortSellCon(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, n, mu, sigma, currentGamma):
        H, a, b, lb, ub, f = gammaShortSellConOptions(currentGamma, n, sigma, mu)

        solver = quadprog(H, f, a, b, lb, ub)
        solverArr = np.asarray(solver)

        return solverArr
