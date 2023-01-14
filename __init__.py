#!/usr/bin/env python3

import numpy as np

from src.app import App

from src.models.noGamma.ew import EqualWeight
from src.models.noGamma.JagannathanMa import JagannathanMa
from src.models.noGamma.kanZhouEw import KanZhouEw
from src.models.noGamma.minVar import MinVar
from src.models.noGamma.minVarShortSellCon import MinVarShortSellCon

from src.models.gamma.meanVar import MeanVar
from src.models.gamma.meanVarShortSellCon import MeanVarShortSellCon
from src.models.gamma.kanZhou import KanZhou
from src.models.gamma.bayesStein import BayesStein
from src.models.gamma.bayesSteinShortSellCon import BayesSteinShortSellCon
from src.models.gamma.macKinlayPastor import MacKinlayPastor

PATH_OLD_SPSECTOR = "data/old/SPSectors.txt"
PATH_OLD_INDUSTRY = "data/old/Industry_noFactors.txt"
PATH_OLD_Mkt_SMB_HML = "data/old/F-F_Research_Data_Factors.txt"

PATH_SPSECTOR = "data/new/processed/sp_sector.csv"
PATH_INDUSTRY = "data/new/processed/industry.csv"
PATH_INTERNATIONAL = "data/new/processed/international_factor.csv"
PATH_Mkt_SMB_HML = "data/new/processed/ff_4_factor.csv"
PATH_FF1 = "data/new/processed/25_portfolios_1_factor.csv"
PATH_FF3 = "data/new/processed/25_portfolios_3_factor.csv"
PATH_FF4 = "data/new/processed/25_portfolios_4_factor.csv"

# MODEL CONSTANTS

# Risk averse levels
GAMMAS = [1, 2, 3, 4, 5, 10]

OMEGAS = []

# Time horizons
TIME_HORIZON = [60, 120]

benchmark = EqualWeight("Equal Weight")
minVar = MinVar("Minimum Variance")
JagannathanMa = JagannathanMa("Jagannathan Ma")
minVarShortSellCon = MinVarShortSellCon("Minimum Variance with Short Sell Constrains")
kanZhouEw = KanZhouEw("Kan Zhou EW")

meanVar = MeanVar("Mean Variance (Markowitz)")
meanVarShortSellCon = MeanVarShortSellCon("Mean Variance with Short Sell Constrains")
kanZhou = KanZhou("Kan Zhou Three Fund")
bayesStein = BayesStein("Bayes Stein")
bayesSteinShortSellCon = BayesSteinShortSellCon("Bayes Stein with Short Sell Constrains")
macKinlayPastor = MacKinlayPastor("MacKinlay and Pastor")

models = [
    benchmark,
    minVar,
    JagannathanMa,
    minVarShortSellCon,
    kanZhouEw,
    meanVar,
    meanVarShortSellCon,
    kanZhou,
    bayesStein,
    bayesSteinShortSellCon,
    macKinlayPastor
]

def main() -> None:
    # app = App(PATH_OLD_SPSECTOR, GAMMAS, OMEGAS, TIME_HORIZON, models,
    #           delim="\s+", date=True, riskFactorPositions=[0])
    # app = App(PATH_OLD_INDUSTRY, GAMMAS, OMEGAS, TIME_HORIZON, models,
    #           delim="\s+", date=True, riskFactorPositions=[0])
    # app = App(PATH_OLD_Mkt_SMB_HML, GAMMAS, OMEGAS, TIME_HORIZON, models,
    #           delim="\s+", date=True, riskFactorPositions=[-1])

    # app = App(PATH_SPSECTOR, GAMMAS, OMEGAS, TIME_HORIZON,
    #           models, riskFactorPositions=[-1])
    # app = App(PATH_INDUSTRY, GAMMAS, OMEGAS, TIME_HORIZON,
    #           models, riskFactorPositions=[-1], riskFreePosition=0, date=True)
    # app = App(PATH_Mkt_SMB_HML, GAMMAS, OMEGAS, TIME_HORIZON,
    #           models, date = True, riskFactorPositions=[0], riskFreePosition=-1)
    # app = App(PATH_FF1, GAMMAS, OMEGAS, TIME_HORIZON,
    #           models, date=True, riskFactorPositions=[-1], riskFreePosition=0)
    app = App(PATH_INTERNATIONAL, GAMMAS, OMEGAS, TIME_HORIZON,
                models, date=True, riskFactorPositions=[-1], riskFreePosition=0)

    sr = app.getSharpeRatios()
    sig = app.getStatisticalSignificanceWRTBenchmark(benchmark)

    print("Sharpe Ratios")

    for key, value in sr.items():
        print("{}: {}".format(key, round(value[0], 5)))

    print()
    print()

    print("Statistical Significances")

    for key, value in sig.items():
        if isinstance(value, np.ndarray):
            print("{}: {}".format(key, round(value[0], 5)))
        else:
            print("{}: {}".format(key, round(value, 5)))
        

if __name__ == "__main__":
    main()