import numpy as np

DATE = 0
OPEN = 1
ADJ_CLOSE = 5 

def readData(filename):
    # read csv file
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    return data

def percentageChange(x, y):
    # return percentage change
    return (y - x) / x