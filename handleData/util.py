import numpy as np

def readData(filename, sepHeader=False):
    if not sepHeader:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)

        return data
    else:
        header = np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=1)
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        
        return [data, header]


def percentageChange(open, close):
    # return percentage change as a percentage
    return (close - open) / open
