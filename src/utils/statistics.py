import numpy as np

"""
    Computes the Sharpe ratio

    param x : [1, rows - M] array, holds the out of sample return values

    returns : real number
"""

def sharpeRato(x):
    # convert to numpy array
    x = np.array(x)
    mean = np.mean(x.T)
    std = np.std(x.T, ddof = 1)
    
    if (abs(mean) > pow(10, -16)):
        sr = mean / std;
    else:
        sr = None
            
    return sr