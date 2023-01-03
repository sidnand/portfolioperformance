import inspect
import numpy as np

from src.utils.statistics import *
from src.utils.filter import *


class Model():
    def __init__(self, name):
        self.name = name

        self.weights = None
        self.weightsBuyHold = None
        self.outSample = None

        self.riskFreeReturns = None
        self.riskyReturns = None

    def init(self, nRisky, m, timeHorizon, riskFreeReturns, riskyReturns):
        self.weights = np.empty((nRisky, m - timeHorizon[-1]))
        self.weightsBuyHold = np.empty(
            (nRisky, m - timeHorizon[-1]))
        self.outSample = np.empty((1, m - timeHorizon[-1]))

        self.riskFreeReturns = riskFreeReturns
        self.riskyReturns = riskyReturns

    def alpha(**kwargs):
        raise NotImplementedError

    def statisticalSignificance(self, benchmark, nSubsets):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError

    def sharpeRatio(self):
        return sharpeRato(self.outSample)

    def statisticalSignificance(self, **kwargs):
        raise NotImplementedError

    def buyHold(self, weights, currentSubset, period):
        a = (1 - sum(weights)) * (1 + self.riskFreeReturns[period + currentSubset])
        b = (1 + (self.riskyReturns[period + currentSubset, :].T +
             self.riskFreeReturns[period + currentSubset]))[np.newaxis].T
        trp = a + weights[np.newaxis].dot(b)

        return ((weights * (1 + (self.riskyReturns[period + currentSubset, :]).T + self.riskFreeReturns[period + currentSubset])) / trp)

    def outOfSampleReturns(self, weights, currentSubset, period):
        return weights.T.dot(self.riskyReturns[period + currentSubset, :][np.newaxis].T)

# Models that do not use gamma
class ModelNoGamma(Model):
    def __init__(self, name):
        super().__init__(name)

    def run(self, params):
        currentSubset = params['currentSubset']
        period = params['period']
        nSubsets = params['nSubsets']

        filter = filterParams(params, self, "alpha")

        alpha = self.alpha(**filter)

        self.weights[:, currentSubset] = alpha[:, 0]

        if currentSubset == 0:
            self.weightsBuyHold[:, currentSubset] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset] = self.buyHold(
                self.weights[:, currentSubset - 1], currentSubset, period)

        if (nSubsets > 1):
            self.outSample[:, currentSubset] = self.outOfSampleReturns(
                alpha, currentSubset, period)[:, 0]

    def statisticalSignificance(self, benchmark, nSubsets):
        z = jobsonKorkieZStat(benchmark, self.outSample, nSubsets)
        p = pval(z)

        return p

# Models that use gamma
class ModelGamma(Model):
    def __init__(self, name):
        super().__init__(name)

    def run(self, params):
        currentSubset = params['currentSubset']
        period = params['period']
        gammas = params['gammas']

        filter = filterParams(params, self, "alpha")

        for currentGamma in gammas:
            
            filter['currentGamma'] = currentGamma

            alpha = self.alpha(**filter)
            self.weights[:, currentSubset] = alpha[:, 0]
            self.outSample[:, currentSubset] = self.outOfSampleReturns(
                alpha, currentSubset, period)[:, 0]

        if currentSubset == 0:
            self.weightsBuyHold[:, currentSubset] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset] = self.buyHold(
                self.weights[:, currentSubset - 1], currentSubset, period)

    def statisticalSignificance(self, benchmark, nSubsets, gammas):
        n = len(gammas)
        z = []

        for i in range(0, n):
            z[i] = jobsonKorkieZStat(benchmark, self.outSample[i, :], nSubsets)

        p = pval(z)

        return p