import sys
sys.path.insert(1, "./src")

import numpy as np
import pandas as pd

from tkinter import *
from tkinter import ttk

from models import *

PATH = "data/SPSectors.txt"

# MODEL PARAMETERS

# how risk averse an investor is, gamma >= 0
GAMMA = [1, 2, 3, 4, 5, 10]

# estimation window; how long we will estimate for
WINDOW = 120

SPSectorsPandas = pd.read_csv(PATH, delim_whitespace = True)

# clean data by removing the date column
SPSectors = SPSectorsPandas.to_numpy()[:, 1:]

rows, cols = SPSectors.shape
n = cols - 1

# portfolio policies
pf = {
    "ew" : np.empty((n, rows - WINDOW)),
    "mv" : np.empty((n, rows - WINDOW))
}

# portfolio weights before rebalancing
pfBuyHold = {
    "ew" : np.empty((n, rows - WINDOW)),
    "mv" : np.empty((n, rows - WINDOW))
}

# out of sample returns
outSample = {
    "ew" : np.empty((1, rows - WINDOW)),
    "mv" : np.empty((1, rows - WINDOW))
}

riskFreeReturns = SPSectors[:, 0] # risk-free asset column
riskyReturns = SPSectors[:, 1:cols] # risky asset column, includes risk factor

def main():
    root = Tk()
    frm = ttk.Frame(root, padding = 10)
    frm.grid()

    root.title("Testing Portfolio Optimization Methods")
    root.geometry('600x400+50+50')

    root.mainloop()

    T = len(riskyReturns) # time period
    nSubsets = 1 if WINDOW == T else T - WINDOW # if WINDOW is the same as time period, then we only have 1 subset

    for shift in range(0, nSubsets):

        riskySubset = riskyReturns[shift:WINDOW + shift, :]
        riskFreeSubset = riskFreeReturns[shift:WINDOW + shift]
        subset = np.column_stack((riskFreeSubset, riskySubset))
        
        mu = np.array([np.mean(riskFreeSubset)])
        mu = np.append(mu, np.vstack(riskySubset.mean(axis = 0)))
        
        totalSigma = np.cov(subset.T)
        sigma = (WINDOW - 1) / (WINDOW - n - 1 - 2) * np.cov(riskySubset.T)
        
        sigmaMLE = (WINDOW - 1) / WINDOW * np.cov(riskySubset.T)
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

    param w : [n, row - WINDOW] array, holds portfolio weights of a specific policy
    param j : integer value, represents current shift position

"""
def buyHold(w, j):

    a = (1 - sum(w)) * (1 + riskFreeReturns[WINDOW + j])
    b = (1 + (riskyReturns[WINDOW + j, :].T + riskFreeReturns[WINDOW + j]))[np.newaxis].T
    trp = a + w[np.newaxis].dot(b)
    
    return ((w * (1 + (riskyReturns[WINDOW + j, :]).T + riskFreeReturns[WINDOW + j])) / trp)

"""

    Computes the out of sample returns

    param w : [n, row - WINDOW] array, holds portfolio weights of a specific policy
    param j : integer value, represents current shift position

"""
def outOfSampleReturns(w, j):
    return w.T.dot(riskyReturns[WINDOW + j, :][np.newaxis].T)

"""

    Computes the Sharpe ratio

    param x : [1, rows - WINDOW] array, holds the out of sample return values

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