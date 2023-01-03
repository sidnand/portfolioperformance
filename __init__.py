#!/usr/bin/env python3

from src.app import App

from src.models.noGamma.ew import EqualWeight
from src.models.noGamma.JagannathanMa import JagannathanMa
from src.models.noGamma.kanZhouEw import KanZhouEw
from src.models.noGamma.minVar import MinVar
from src.models.noGamma.minVarShortSellCon import MinVarShortSellCon

from src.models.gamma.meanVar import MeanVar
from src.models.gamma.meanVarShortSellCon import MeanVarShortSellCon
from src.models.gamma.bayesStein import BayesStein
from src.models.gamma.bayesSteinShortSellCon import BayesSteinShortSellCon

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
    BayesStein("Bayes Stein"),
    BayesSteinShortSellCon("Bayes Stein with Short Sell Constrains")
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