# OPTIONS SHARED BY MULTIPLE MODELS

import numpy as np

def minVarConOptions(s):
    n = s.shape[1]
    ub = np.ones((1, n))
    aeq = np.ones((1, n))
    beq = [1]
    f = np.zeros((n, 1))

    return n, ub, aeq, beq, f