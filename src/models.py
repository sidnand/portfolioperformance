# CODE FOR ALL THE PORTFOLIO MODELS
import numpy as np
from .utils import quadprog

"""
Computes the weight of each asset in the portfolio, where each asset weight is 1/N,
N is the number of assets in the portfolio

param n : number of columns (assets)
returns : [n - 1, 1] array, where each element is 1/N
"""


def ew(n):
   return 1/n * np.ones((n - 1, 1))


"""
Computes the minimum variance portfolio weights
To learn more: https://www.youtube.com/watch?v=krm6upwkg0M

param invSigmaMLE : the inverse of the MLE of the covariance matrix
param AMLE : integer value which is equal to invSigmaMLE * invSigmaMLE transpose
param n : number of columns (assets)
returns : [n - 1, 1] array of the minimum variance portfolio weights
"""


def minVar(invSigmaMLE, AMLE, n):
   return 1/AMLE * invSigmaMLE.dot(np.ones((n - 1, 1)))


"""
Computes the minimum variance portfolio weights with short sell constraints
This uses quadratic programming: https://en.wikipedia.org/wiki/Quadratic_programming
To learn more: https://www.youtube.com/watch?v=oaiiyIsbNdI, https://www.youtube.com/watch?v=GZb9647X8sg

param sigmaMLE : MLE of the covariance matrix

"""
def minVarShortSellCon(sigmaMLE):

   m, n = sigmaMLE.shape
   ub = np.ones((1, n))
   aeq = np.ones((1, n))
   beq = [1]
   lb = np.zeros((1, n))
   f = np.zeros((n, 1))

   solver = quadprog(sigmaMLE, f, aeq, beq, lb, ub)
   solverArr = np.asarray(solver)

   return solverArr


"""
Computes the minimum variance portfolio weights with constraints
This uses quadratic programming: https://en.wikipedia.org/wiki/Quadratic_programming
To learn more: https://www.youtube.com/watch?v=oaiiyIsbNdI, https://www.youtube.com/watch?v=GZb9647X8sg

param sigmaMLE : MLE of the covariance matrix

"""
def jagannathanMa(sigma):
   m, n = sigma.shape
   ub = np.ones((1, n))
   aeq = np.ones((1, n))
   beq = [1]
   lb = np.ones((1, n)) / (2 * n)
   f = np.zeros((n, 1))

   solver = quadprog(sigma, f, aeq, beq, lb, ub)
   solverArr = np.asarray(solver)

   return solverArr

def kanZhouEw(N, M, sigma):
   invSigmaMLE = np.linalg.inv(sigma)

   esige = (np.ones((1, N)) @ sigma @ np.ones((N, 1)))[0][0]
   einvsige = (np.ones((1, N)) @ invSigmaMLE @ np.ones((N, 1)))[0][0]

   k = (M^2 * (M-2)) / ((M-N-1) * (M-N-2) * (M-N-4))

   d = ((M - N - 2) * esige * einvsige - N**2 * M)/(N**2 * (M - N - 2) * k * einvsige - 2 * M * N**2 * einvsige + (M - N - 2) * einvsige**2 * esige)
   c = 1 - d * einvsige

   # alpha = c * 1 / N * np.ones((N,1)) + d * invSigmaMLE @ np.ones((N,1))

   alpha = (c * (1 / N) * np.ones((N,1))) + ((d * invSigmaMLE) @ np.ones((N,1)))

   return alpha

def meanVariance(gamma, invSigmaMLE, mu):
   return (1/gamma) * invSigmaMLE * mu[2:]