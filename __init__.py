#!/usr/bin/env python3

from src.app import App

from src.models.ew import EqualWeight
from src.models.JagannathanMa import JagannathanMa
from src.models.kanZhouEw import KanZhouEw
from src.models.minVar import MinVar
from src.models.minVarShortSellCon import MinVarShortSellCon

from src.models.meanVar import MeanVar
from src.models.meanVarShortSellCon import MeanVarShortSellCon

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
    MinVar("Minimum Variance"),
    JagannathanMa("Jagannathan Ma"),
    MinVarShortSellCon("Minimum Variance with Short Sell Constrains"),
    KanZhouEw("Kan Zhou EW"),

    MeanVar("Mean Variance (Markowitz)"),
    MeanVarShortSellCon("Mean Variance with Short Sell Constrains"),
]

def main() -> None:
    app = App(PATH, GAMMAS, TIME_HORIZON, models)
    # app = App(PATH_OLD, GAMMAS, TIME_HORIZON, models, delim="\s+", date=True)

    sr = app.getSharpeRatios()
    sig = app.getStatisticalSignificances(benchmark)

    print("Sharpe Ratios")

    for key, value in sr.items():
        print("{}: {}".format(key, value))

    print()
    print()

    print("Statistical Significances")

    for key, value in sig.items():
        print("{}: {}".format(key, value))

if __name__ == "__main__":
    main()