# CODE FOR ALL THE PORTFOLIO MODELS

import numpy as np

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

   param invSigmaMLE : the inverse of the MLE of the covariance matrix
   param AMLE : integer value which is equal to invSigmaMLE * invSigmaMLE transpose
   param n : number of columns (assets)
   returns : [n - 1, 1] array of the minimum variance portfolio weights
"""
def minVar(invSigmaMLE, AMLE, n):
   return 1/AMLE * invSigmaMLE.dot(np.ones((n - 1, 1)))