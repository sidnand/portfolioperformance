import numpy as np
import scipy.stats as stats
from math import sqrt

def se(x):
    return np.std(x.T, ddof = 1) / np.sqrt(x.shape[0])

def jobsonKorkieZStat(benchmark, outSample, nSubsets):
    mu1 = np.mean(benchmark)
    mu2 = np.mean(outSample)

    sigma = np.cov(benchmark, outSample)

    sigma1 = np.sqrt(sigma[0,0])
    sigma2 = np.sqrt(sigma[1,1])
    sigma12 = sigma[0,1]

    a = 2 * sigma1**2 * sigma2**2
    b = 2 * sigma1 * sigma2 * sigma12
    c = (1/2) * mu1**2 * sigma2**2
    d = (1/2) * mu2**2 * sigma1**2
    e = mu1 * mu2 / (sigma1 * sigma2) * sigma12**2
    f = (a - b + c + d - e)

    theta = (1/nSubsets) * f

    if (theta <= 0): return (sigma2 * mu1 - sigma1 * mu2) / np.finfo(float).eps
    else: return (sigma2 * mu1 - sigma1 * mu2) / sqrt(theta)
    

def sharpeRato(x):
    meanRet = np.mean(x)
    stdRet = np.std(x)
    
    if (abs(meanRet) > pow(10, -16)):
        sr = meanRet / stdRet;
    else:
        sr = None
            
    return sr

def pval(z):
    return 1 - stats.norm.cdf(z)