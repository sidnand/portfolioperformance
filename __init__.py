#!/usr/bin/env python3

import pandas as pd
import random

from src.system import *

# MODEL CONSTANTS

PATH = "data/SPSectors.txt"

# how risk averse an investor is, gamma >= 0
GAMMA = [1, 2, 3, 4, 5, 10]

# estimation window; how long we will estimate for
T = 120

SPSectorsPandas = pd.read_csv(PATH, delim_whitespace = True)

# clean data by removing the date column
SPSectors = SPSectorsPandas.to_numpy()[:, 1:]

COLS = SPSectors.shape[1]

def main():
    SYSTEM = System(SPSectors, T)

    sr = SYSTEM.getSharpeRatios(GAMMA)

    for key, value in sr.items():
        print("{}: {}".format(key, round(value, 4)))

if __name__ == "__main__":
    main()