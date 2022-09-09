import sys
sys.path.insert(1, "./src")

import numpy as np
import pandas as pd

from tkinter import *

from models import *
from ui import *

# UI CONSTANTS

TITLE = "Testing Portfolio Optimization Methods"
WIDTH = 600
HEIGHT = 400

WINDOW = UI(Tk(), TITLE, WIDTH, HEIGHT)

# MODEL PARAMETERS

PATH = "data/SPSectors.txt"

# how risk averse an investor is, gamma >= 0
GAMMA = [1, 2, 3, 4, 5, 10]

# estimation window; how long we will estimate for
M = 120




SPSectorsPandas = pd.read_csv(PATH, delim_whitespace = True)

# clean data by removing the date column
SPSectors = SPSectorsPandas.to_numpy()[:, 1:]

rows, cols = SPSectors.shape
n = cols - 1

# portfolio policies
pf = {
    "ew" : np.empty((n, rows - M)),
    "mv" : np.empty((n, rows - M))
}

# portfolio weights before rebalancing
pfBuyHold = {
    "ew" : np.empty((n, rows - M)),
    "mv" : np.empty((n, rows - M))
}

# out of sample returns
outSample = {
    "ew" : np.empty((1, rows - M)),
    "mv" : np.empty((1, rows - M))
}

riskFreeReturns = SPSectors[:, 0] # risk-free asset column
riskyReturns = SPSectors[:, 1:cols] # risky asset column, includes risk factor

def main():
    WINDOW.show()

    T = len(riskyReturns) # time period
    nSubsets = 1 if M == T else T - M # if M is the same as time period, then we only have 1 subset

    for shift in range(0, nSubsets):

        riskySubset = riskyReturns[shift:M + shift, :]
        riskFreeSubset = riskFreeReturns[shift:M + shift]
        subset = np.column_stack((riskFreeSubset, riskySubset))
        
        mu = np.array([np.mean(riskFreeSubset)])
        mu = np.append(mu, np.vstack(riskySubset.mean(axis = 0)))
        
        totalSigma = np.cov(subset.T)
        sigma = (M - 1) / (M - n - 1 - 2) * np.cov(riskySubset.T)
        
        sigmaMLE = (M - 1) / M * np.cov(riskySubset.T)
        invSigmaMLE = np.linalg.inv(sigmaMLE)

        AMLE = np.ones((1, cols - 1)).dot(invSigmaMLE).dot(np.ones((cols - 1, 1)))
        alphaMV = minVar(invSigmaMLE, AMLE, cols)
        
        # 1/N
        alphaTew = ew(cols)
        pf["ew"][:, shift] = alphaTew[:, 0]
        
        # mean-variance
        alphaMV = minVar(invSigmaMLE, AMLE, cols)
        pf["mv"][:, shift] = alphaMV[:, 0]

        # buy and hold
        if shift == 0:
            pfBuyHold["ew"][:, shift]= alphaTew[:, 0]
            pfBuyHold["mv"][:, shift]= alphaMV[:, 0]
        else:
            pfBuyHold["ew"][:, shift] = buyHold(pf["ew"][:, shift - 1], shift)
            pfBuyHold["mv"][:, shift] = buyHold(pf["mv"][:, shift - 1], shift)
            
        if (nSubsets > 1):
        # out of sample returns
            outSample["ew"][:, shift] = outOfSampleReturns(alphaTew, shift)[:, 0]
            outSample["mv"][:, shift] = outOfSampleReturns(alphaMV, shift)[:, 0]

    sr = sharpeRato(outSample["ew"])
    mv = sharpeRato(outSample["mv"])

    print(sr)
    print(mv)

"""

    Computes a new portfolio weight after a shift

    param w : [n, row - M] array, holds portfolio weights of a specific policy
    param j : integer value, represents current shift position

"""
def buyHold(w, j):

    a = (1 - sum(w)) * (1 + riskFreeReturns[M + j])
    b = (1 + (riskyReturns[M + j, :].T + riskFreeReturns[M + j]))[np.newaxis].T
    trp = a + w[np.newaxis].dot(b)
    
    return ((w * (1 + (riskyReturns[M + j, :]).T + riskFreeReturns[M + j])) / trp)

"""

    Computes the out of sample returns

    param w : [n, row - M] array, holds portfolio weights of a specific policy
    param j : integer value, represents current shift position

"""
def outOfSampleReturns(w, j):
    return w.T.dot(riskyReturns[M + j, :][np.newaxis].T)

"""

    Computes the Sharpe ratio

    param x : [1, rows - M] array, holds the out of sample return values

"""
def sharpeRato(x):
    mean = np.mean(x.T)
    std = np.std(x.T, ddof = 1)
    
    if (abs(mean) > pow(10, -16)):
        sr = mean / std;
    else:
        sr = None
            
    return sr

if __name__ == "__main__":
    main()