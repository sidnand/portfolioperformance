import sys
sys.path.insert(1, "./src")

import numpy as np
import pandas as pd

from tkinter import *

from ui import *
from system import *
from enum_policy import *

# UI CONSTANTS

TITLE = "Testing Portfolio Optimization Methods"
WIDTH = 600
HEIGHT = 400

WINDOW = UI(Tk(), TITLE, WIDTH, HEIGHT)

# MODEL CONSTANTS

PATH = "data/SPSectors.txt"

PF = ["ew", "minvar", "minvar-constrained"]

# how risk averse an investor is, gamma >= 0
GAMMA = [1, 2, 3, 4, 5, 10]

# estimation window; how long we will estimate for
M = 120


SPSectorsPandas = pd.read_csv(PATH, delim_whitespace = True)

# clean data by removing the date column
SPSectors = SPSectorsPandas.to_numpy()[:, 1:]

COLS = SPSectors.shape[1]

riskFreeReturns = SPSectors[:, 0] # risk-free asset column
riskyReturns = SPSectors[:, 1:COLS] # risky asset column, includes risk factor

def main():
    SYSTEM = System(riskFreeReturns, riskyReturns, M)

    sr = SYSTEM.getSharpeRatios()
    
    keys = np.array([list(sr.keys())])
    values = np.array([list(sr.values())])
    data = np.vstack((keys, values))

    WINDOW.table(data)

    WINDOW.show()

if __name__ == "__main__":
    main()