import numpy as np

from .utils.statistics import *

class System:
    def __init__(self, data, timeHorizon, models, gamma):
        (self.m, self.n) = data.shape

        self.riskFreeReturns = data[:, 0] # risk-free asset column
        self.riskyReturns = data[:, 1:self.n] # risky asset column, includes risk factor
        self.timeHorizon = timeHorizon # estimation window length

        self.models = models
        self.gamma = gamma

        self.nRisky = self.n - 1 # number of risky variables
        self.T = len(self.riskyReturns) # time period
        self.upperM = self.timeHorizon[-1] # upper bound of time horizon


        self.weights = dict.fromkeys(
                [model.name for model in self.models],
                np.empty((self.nRisky, self.m - self.timeHorizon[-1]))
            )
        
        self.weightsBuyHold = dict.fromkeys(
                [model.name for model in self.models],
                np.empty((self.nRisky, self.m - self.timeHorizon[-1]))
            )

        self.outSample = dict.fromkeys(
                [model.name for model in self.models],
                np.empty((1, self.m - self.timeHorizon[-1]))
            )

    def run(self):
        for k in self.timeHorizon:
            m = k # current time horizon
            shift = self.upperM - m # shift in time horizon
            m = m + shift # update time horizon

            nSubsets = 1 if m == self.T else self.T - m # if m is the same as time period, then we only have 1 subset

            for j in range(0, nSubsets):
                riskySubset = self.riskyReturns[j+shift:m+j-1, :]
                riskFreeSubset = self.riskFreeReturns[j+shift:m+j-1]
                subset = np.column_stack((riskFreeSubset, riskySubset))

                mu = np.append(np.array([
                                    np.mean(riskFreeSubset)
                                ]),
                                np.vstack(riskySubset.mean(axis = 0)))

                totalSigma = np.cov(subset.T) # variance covariance matrix
                sigma = (m - 1) / (m - self.nRisky - 2) * np.cov(riskySubset.T)
                sigmaMLE = (m - 1) / m * np.cov(riskySubset.T)
                invSigmaMLE = np.linalg.inv(sigmaMLE)
                amle = np.ones((1, self.nRisky)) @ invSigmaMLE @ np.ones((self.nRisky, 1))

                params = {
                    "n": self.n,
                    "sigma": sigma,
                    "sigmaMLE": sigmaMLE,
                    "invSigmaMLE": invSigmaMLE,
                    "amle": amle,
                    "gamma": self.gamma,
                    "nRisky": self.nRisky,
                    "m": m,
                    "mu": mu
                }

                for model in self.models:
                    i = model.name
                    alpha = model.run(params)

                    self.weights[i][:, j] = alpha[:, 0]

                    if j == 0: self.weightsBuyHold[i][:, j] = alpha[:, 0]
                    else: self.weightsBuyHold[i][:, j] = self.buyHold(self.weights[i][:, j - 1], j, m)

                    if nSubsets > 1: self.outSample[i][:, j] = self.outOfSampleReturns(alpha, j, m)[:, 0]

    def getSharpeRatios(self):
        # for model in self.models:
        #     i = model.name
        #     print(self.weights)

        pass

    def buyHold(self, w, j, m):
        a = (1 - sum(w)) * (1 + self.riskFreeReturns[m + j])
        b = (1 + (self.riskyReturns[m + j, :].T + self.riskFreeReturns[m + j]))[np.newaxis].T
        trp = a + w[np.newaxis].dot(b)
        
        return ((w * (1 + (self.riskyReturns[m + j, :]).T + self.riskFreeReturns[m + j])) / trp)

    def outOfSampleReturns(self, w, j, m):
        return w.T.dot(self.riskyReturns[m + j, :][np.newaxis].T)