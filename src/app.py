from typing import *

import pandas as pd
import numpy as np

from .model import Model

'''
    @brief Entry point for the application
'''

class App:
    '''
        Initialize the app with the following parameters:

        @param path: path to the data file
        @param gammas: risk averse levels
        @param omegas: 
        @param timeHorizon: time horizons for rolling window
        @param models: list of models to run
        @param delim: delimiter for the data file, default is ",", other option is "\s+" for spaces and tabs
        @param date: whether the data file has a date column, as the first column, default is False
        @param logScale: whether the data file is in log scale, default is False
        @param riskFactorPos: list of positions of the risk factors in the data file, default is None, start at index 0
        @param windowType: type of window, default is "Rolling", other option is "Increasing"

        @return: App object
    '''

    def __init__(self, path: str, gammas: list[int], omegas: list[int], timeHorizon: list[int], models: list[Model],
                delim: Literal[",", "\s+"] = ",", date: bool = False, logScale: bool = False,
                 riskFactorPositions: list[int] = [], windowType: Literal["Rolling", "Increasing"] = "Rolling") -> None:
        self.path = path
        self.delim = delim
        self.originalData = self.readFile(path, delim, date)

        self.data, self.assetNames = self.getData()
        self.period = self.data.shape[0]

        self.models = models
        self.gammas = gammas
        self.omegas = omegas
        self.timeHorizon = timeHorizon

        # risk-free asset column
        self.riskFreeReturns = self.getRiskFreeReturns(logScale)
        # risky asset column, includes risk factor
        self.riskyReturns, self.withoutRiskFactorReturns = self.getRiskyReturns(logScale, riskFactorPositions)

        self.n = self.riskyReturns.shape[1] + 1

        # number of risky variables
        self.nRisky = self.n - len(riskFactorPositions)
        # total time period
        self.t = len(self.riskyReturns)
        # last time horizon
        self.upperM = self.timeHorizon[-1]

        self.nSubsets = None

        self.initModels()
        self.run()

    '''
        @brief Read the data file into a pandas DataFrame

        @param path: path to the data file
        @param delim: delimiter for the data file, default is ",", other option is "\s+" for spaces and tabs
        @param date: whether the data file has a date column, as the first column, default is False

        @return: pandas DataFrame of the data file
    '''
    def readFile(self, path: str, delim: str, date: bool) -> pd.DataFrame:
        if date:
            return pd.read_csv(path, delimiter=delim).iloc[:, 1:]

        return pd.read_csv(path, delimiter=delim)

    '''
        @brief Get the data from the pandas DataFrame

        @return: numpy array of the data
        @return: list of asset names
    '''
    def getData(self) -> np.ndarray and list[str]:
        # asset names exclude the risk-free asset
        assetNames = list(self.originalData.columns)
        data = self.originalData.to_numpy()

        return data, assetNames

    '''
        @brief Get the risk-free returns

        @param logScale: whether the data file is in log scale, default is False

        @return: numpy array of the risk-free returns
    '''
    def getRiskFreeReturns(self, logScale:bool) -> np.ndarray:
        if logScale:
            return np.exp(self.data[:, 0]) - 1

        return self.data[:, 0]

    '''
        @brief Get the risky returns

        @param logScale: whether the data file is in log scale, default is False
        @param riskFactorPositions: list of positions of the risk factors in the data file, default is None, start at index 0

        @return: numpy array of the risky returns
        @return: numpy array of the risky returns without the risk factors
    '''
    def getRiskyReturns(self, logScale: list, riskFactor: list) -> np.ndarray and np.ndarray:
        withoutRiskFactorReturns = None

        riskyReturns = np.exp(self.data[:, 1:]) - 1 if logScale else self.data[:, 1:]
        if riskFactor:
            withoutRiskFactorReturns = np.exp(np.delete(self.data, riskFactor, 1)) - 1 if logScale else np.delete(self.data, riskFactor, 1)

        return riskyReturns, withoutRiskFactorReturns

    '''
        @brief Initalize the models with the following parameters

        @return: None
    '''
    def initModels(self) -> None:
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

    '''
        @brief Get the statistics of the data needed for the models

        @param riskFreeSubset: numpy array of the risk-free returns
        @param riskySubset: numpy array of the risky returns
        @param subset: numpy array of the returns for the given time period
        @param period: time period

        @return: dictionary of the statistics
    '''
    def getStats(self, riskFreeSubset, riskySubset, subset, period) -> dict[str, np.ndarray or float]:
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

    '''
        @brief Run the models

        @return: None
    '''
    def run(self) -> None:
        # loop through the time horizon, this is the rolling window
        for currentPeriod in self.timeHorizon:
            period = currentPeriod  # current time horizon
            shift = self.upperM - period  # shift in time horizon
            period = currentPeriod + shift  # update time horizon given shift

            # if period is the same as time currentPeriod, then we only have 1 subset
            self.nSubsets = 1 if period == self.t else self.t - period

            for currentSubset in range(0, self.nSubsets):

                # get the subset of the data
                riskySubset = self.riskyReturns[currentSubset +
                                                shift:period+currentSubset-1, :]
                riskFreeSubset = self.riskFreeReturns[currentSubset +
                                                      shift:period+currentSubset-1]

                # combine the risk-free and risky returns
                subset = np.column_stack((riskFreeSubset, riskySubset))

                stats = self.getStats(riskFreeSubset, riskySubset, subset, period)

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

    '''
        @brief Get the sharpe ratios of the models

        @return: dictionary of the sharpe ratios
    '''
    def getSharpeRatios(self) -> dict[str, float]:
        sr = {}

        for model in self.models:
            sr[model.name] = model.sharpeRatio()

        return sr

    '''
        @brief Get the statistical significance w.r.t a benchmark of the models

        @param benchmark: out-of-sample returns for a benchmark model

        @return: dictionary of the statistical significance
    '''
    def getStatisticalSignificanceWRTBenchmark(self, benchmark: np.array) -> dict[str, float]:
        sig = {}
        params = {
            "benchmark": benchmark.outSample,
            "nSubsets": self.nSubsets,
            "gammas": self.gammas
        }

        for model in self.models:
            sig[model.name] = model.statisticalSignificanceWRTBenchmark(params)

        return sig
