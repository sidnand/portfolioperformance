import pandas as pd
import numpy as np


class App:
    def __init__(self, path, gammas, timeHorizon, models, delim=",", date=False) -> None:
        self.path = path
        self.delim = delim
        self.originalData = self.readFile()

        self.data = self.getData(date)
        (self.period, self.n) = self.data.shape

        self.models = models
        self.gammas = gammas
        self.timeHorizon = timeHorizon

        # risk-free asset column
        self.riskFreeReturns = self.data[:, 0]
        # risky asset column, includes risk factor
        self.riskyReturns = self.data[:, 1:self.n]

        # number of risky variables
        self.nRisky = self.n - 1
        # time currentPeriod
        self.t = len(self.riskyReturns)
        # last time horizon
        self.upperM = self.timeHorizon[-1]

        self.nSubsets = None

        self.initModels()
        self.run()

    def readFile(self) -> pd.DataFrame:
        return pd.read_table(self.path, sep=self.delim)

    def getData(self, date) -> np.ndarray:
        withDate = self.originalData.to_numpy()[:, 1:]
        withoutDate = self.originalData.to_numpy()

        return withDate if date else withoutDate

    def initModels(self):
        for model in self.models:
            model.init(self.nRisky, self.period, self.timeHorizon,
                       self.riskFreeReturns, self.riskyReturns)

    def getStats(self, riskFreeSubset, riskySubset, subset, period) -> dict:
        mu = np.append(np.array([np.mean(riskFreeSubset)]),
                       np.vstack(riskySubset.mean(axis=0)))

        totalSigma = np.cov(subset.T)
        sigma = (period - 1) / (period - self.nRisky - 2) * \
            np.cov(riskySubset.T)

        sigmaMLE = (period - 1) / period * np.cov(riskySubset.T)
        invSigmaMLE = np.linalg.inv(sigmaMLE)

        amle = np.ones((1, self.nRisky)
                       ) @ invSigmaMLE @ np.ones((self.nRisky, 1))

        return {
            "mu": mu,
            "sigma": sigma,
            "totalSigma": totalSigma,
            "sigmaMLE": sigmaMLE,
            "invSigmaMLE": invSigmaMLE,
            "amle": amle
        }

    def run(self):
        for currentPeriod in self.timeHorizon:
            period = currentPeriod  # time horizon
            shift = self.upperM - period  # shift in time horizon
            period = currentPeriod + shift  # update time horizon

            # if period is the same as time currentPeriod, then we only have 1 subset
            self.nSubsets = 1 if period == self.t else self.t - period

            for currentSubset in range(0, self.nSubsets):

                riskySubset = self.riskyReturns[currentSubset +
                                                shift:period+currentSubset-1, :]
                riskFreeSubset = self.riskFreeReturns[currentSubset +
                                                      shift:period+currentSubset-1]
                subset = np.column_stack((riskFreeSubset, riskySubset))

                stats = self.getStats(
                    riskFreeSubset, riskySubset, subset, period)

                params = stats | {
                    "n": self.n,
                    "gammas": self.gammas,
                    "nRisky": self.nRisky,

                    "period": period,
                    "currentSubset": currentSubset,
                    "nSubsets": self.nSubsets,
                }

                for model in self.models:
                    model.run(params)

    def getSharpeRatios(self):
        sr = {}

        for model in self.models:
            sr[model.name] = model.sharpeRatio()

        return sr

    def getStatisticalSignificances(self, benchmark):
        sig = {}

        for model in self.models:
            sig[model.name] = model.statisticalSignificance(
                benchmark.outSample, self.nSubsets)

        return sig
