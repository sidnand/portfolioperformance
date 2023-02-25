# OPTIONS SHARED BY MULTIPLE MODELS

import numpy as np

def minVarConOptions(s):
    n = s.shape[1]
    ub = np.ones((1, n))
    aeq = np.ones((1, n))
    beq = [1]
    f = np.zeros((n, 1))

    return n, ub, aeq, beq, f

def gammaShortSellConOptions(currentGamma, n, s, mu):
    mu = mu[1:]

    H = currentGamma * s
    a = np.ones((1, n - 1))
    b = 1
    lb = np.zeros((1, n - 1))
    ub = np.ones((1, n - 1))
    f = -mu.T

    return H, a, b, lb, ub, f
