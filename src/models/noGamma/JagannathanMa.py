import numpy as np

from ...model import *
from ...utils.quadprog import quadprog
from ...utils.sharedOptions import minVarConOptions


class JagannathanMa(ModelNoGamma):
    def __init__(self, name):
        super().__init__(name)

    def alpha(self, sigma):
        n, ub, aeq, beq, f = minVarConOptions(sigma)
        lb = np.ones((1, n)) / (2 * n)

        solver = quadprog(sigma, f, aeq, beq, lb, ub)
        solverArr = np.asarray(solver)

        return solverArr