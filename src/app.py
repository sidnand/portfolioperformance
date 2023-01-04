import pandas as pd
import numpy as np


class App:
    def __init__(self, path, gammas, timeHorizon, models, delim=",", date=False) -> None:
        self.path = path
        self.delim = delim
        self.originalData = self.readFile()

        self.data, self.assetNames = self.getData(date)
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
        assetNames = None
        data = None

        if date:
            assetNames = list(self.originalData.columns[2:])
            data = self.originalData.to_numpy()[:, 1:]
        else:
            assetNames = list(self.originalData.columns[1:])
            data = self.originalData.to_numpy()

        return data, assetNames

    def initModels(self):
        params = {
            "nRisky": self.nRisky,
            "period": self.period,
            "timeHorizon": self.timeHorizon,
            "riskFreeReturns": self.riskFreeReturns,
            "riskyReturns": self.riskyReturns,
            "gammas": self.gammas,
            "assetNames": self.assetNames,
        }

        for model in self.models:
            model.init(params)

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

        Y = np.expand_dims(mu[1:], axis=1)
        sigmaHat = (period - 1) / (period - self.nRisky - 2) * np.cov(riskySubset.T)
        invSigmaHat = np.linalg.inv(sigmaHat)
        Ahat = np.ones((1, self.nRisky)) @ invSigmaHat @ np.ones((self.nRisky, 1))
        Y0 = (np.ones((1, self.nRisky)) @ invSigmaHat @ Y) / Ahat
        w = (self.nRisky + 2) / ((self.nRisky + 2) + (Y - Y0).T @ (period * invSigmaHat) @ (Y - Y0))
        lamda = (self.nRisky + 2) / ((Y - Y0).T @ invSigmaHat @ (Y - Y0))
        muBS = np.append(np.array([np.mean(riskFreeSubset)]), (1 - w) * Y + w * Y0)
        sigmaBS = sigmaHat * (1 + 1 / (period + lamda)) + lamda / (period * (period + 1 + lamda)) * np.ones((self.nRisky, 1)) @ np.ones((1, self.nRisky)) / Ahat
        invSigmaBS = np.linalg.inv(sigmaBS)
        totalSigmaBS = (period - 1) / (period - self.nRisky - 2) * totalSigma

        return {
            "mu": mu,
            "sigma": sigma,
            "totalSigma": totalSigma,
            "sigmaMLE": sigmaMLE,
            "invSigmaMLE": invSigmaMLE,
            "amle": amle,

            "Y": Y,
            "sigmaHat": sigmaHat,
            "invSigmaHat": invSigmaHat,
            "Ahat": Ahat,
            "Y0": Y0,
            "w": w,
            "lamda": lamda,
            "muBS": muBS,
            "sigmaBS": sigmaBS,
            "invSigmaBS": invSigmaBS,
            "totalSigmaBS": totalSigmaBS
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
                    model.runOutSample(params)

    def getSharpeRatios(self):
        sr = {}

        for model in self.models:
            sr[model.name] = model.sharpeRatio()

        return sr

    def getStatisticalSignificances(self, benchmark):
        sig = {}
        params = {
            "benchmark": benchmark.outSample,
            "nSubsets": self.nSubsets,
            "gammas": self.gammas
        }

        for model in self.models:
            sig[model.name] = model.statisticalSignificance(params)

        return sig
