import numpy as np

from ...model import *
from ...utils.quadprog import *


class MeanVarShortSellCon(ModelGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, n, mu, sigma, currentGamma):

        mu = mu[1:]

        H = currentGamma * sigma
        a = np.ones((1, n - 1))
        b = 1
        lb = np.zeros((1, n - 1))
        ub = np.ones((1, n - 1))
        f = -mu.T

        solver = quadprog(H, f, a, b, lb, ub)
        solverArr = np.asarray(solver)

        return solverArr
