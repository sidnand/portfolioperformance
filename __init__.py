#!/usr/bin/env python3

import pandas as pd

from src.system import *

# MODEL CONSTANTS

PATH_OLD = "data/old/SPSectors.txt"
PATH = "data/new/processed/sp_sector.csv"

# how risk averse an investor is, gamma >= 0
GAMMA = [1, 2, 3, 4, 5, 10]

# estimation window; how long we will estimate for
T = 120

SPSectorsPandas = pd.read_csv(PATH)
SPSectorsPandasOld = pd.read_csv(PATH_OLD, delim_whitespace = True)

SPSectors = SPSectorsPandas.to_numpy()
SPSectorsOld = SPSectorsPandasOld.to_numpy()[:, 1:]

COLS = SPSectors.shape[1]
COLS_OLD = SPSectorsOld.shape[1]

def main():
    SYSTEM = System(SPSectors, T)
    # SYSTEM = System(SPSectorsOld, T)

    sr = SYSTEM.getSharpeRatios(GAMMA)

    for key, value in sr.items():
        print("{}: {}".format(key, round(value, 4)))

if __name__ == "__main__":
    main()

# SPSectorsPandas_old = pd.read_csv(PATH_OLD, delim_whitespace = True)
# SPSectors_old = SPSectorsPandas_old.to_numpy()

# SPSectorsPandas_new = pd.read_csv(PATH, delim_whitespace = True)
# SPSectors_new = SPSectorsPandas_new.to_numpy()