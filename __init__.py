#!/usr/bin/env python3

from src.app import App

from src.models.ew import EqualWeight
from src.models.JagannathanMa import JagannathanMa
from src.models.kanZhouEw import KanZhouEw
from src.models.meanVar import MeanVar
from src.models.minVar import MinVar
from src.models.minVarShortSellCon import MinVarShortSellCon

PATH = "data/new/processed/sp_sector.csv"
PATH_OLD = "data/old/SPSectors.txt"

# MODEL CONSTANTS

# Risk averse levels
GAMMAS = [1, 2, 3, 4, 5, 10]

# Time horizons
TIME_HORIZON = [60, 120]

benchmark = EqualWeight("Equal Weight")

models = [
    benchmark,
    JagannathanMa("Jagannathan Ma"),
    MinVar("Minimum Variance"),
    MinVarShortSellCon("Minimum Variance Short Sell Constrained"),
    KanZhouEw("Kan Zhou EW"),
    MeanVar("Mean Variance (Markowitz)")
]

def main() -> None:
    # app = App(PATH, GAMMA, TIME_HORIZON, models)
    app = App(PATH_OLD, GAMMAS, TIME_HORIZON, models, delim="\s+", date=True)

    sr = app.getSharpeRatios()
    sig = app.getStatisticalSignificances(benchmark)

    for key, value in sr.items():
        print("{}: {}".format(key, value))

    for key, value in sig.items():
        print("{}: {}".format(key, value))

if __name__ == "__main__":
    main()