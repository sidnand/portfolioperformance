import numpy as np
import pandas as pd

from src.utils.statistics import *
from src.utils.filter import *


class Model():
    def __init__(self, name):
        self.name = name

        self.assetNames = None

        self.weights = None
        self.weightsBuyHold = None
        self.outSample = None

        self.riskFreeReturns = None
        self.riskyReturns = None

    def init(self, params):
        filter = filterParams(params, self, "_init")

        self.assetNames = params['assetNames']

        try:
            self._init(**filter)
        except:
            raise NotImplementedError("Model does not implement _init method")

    def alpha(**kwargs):
        raise NotImplementedError("Model does not implement alpha method")

    def runOutSample(self, params):
        raise NotImplementedError("Model does not implement runOutSample method")

    def runInSample(self, params):
        pass
        # raise NotImplementedError("Model does not implement runInSample method")

    def sharpeRatio(self):
        return sharpeRato(self.outSample)

    def statisticalSignificanceWRTBenchmark(self, params):
        filter = filterParams(params, self, "_statisticalSignificanceWRTBenchmark")
        
        try:
            return self._statisticalSignificanceWRTBenchmark(**filter)
        except:
            raise NotImplementedError("Model does not implement _statisticalSignificanceWRTBenchmark method")

    def statisticalSignificanceSR0(self, sr, gammas = None):
        if not gammas:
            z = zSharpeRatio0(self.outSample, sr)
            p = pValue(z)

            return p
        else:
            z = list(range(len(gammas)))

            for i in range(0, len(gammas)):
                z[i] = zSharpeRatio0(self.outSample[i], sr)
            
            p = pValue(z)

            return p

    def toDataFrame(self, gammas = None):
        if not gammas:
            weights = pd.DataFrame(self.weights.T)
            weights.columns = self.assetNames

            weightsBuyHold = pd.DataFrame(self.weightsBuyHold.T)
            weightsBuyHold.columns = self.assetNames

            outSample = pd.DataFrame(self.outSample.T)
            outSample.columns = ["Out of Sample"]

            return weights, weightsBuyHold, outSample

        else:
            weights = [None] * len(gammas)
            weightsBuyHold = [None] * len(gammas)
            outSample = [None] * len(gammas)

            for i in range(0, len(gammas)):
                weights[i] = pd.DataFrame(self.weights[:, :, i].T)
                weights[i].columns = self.assetNames

                weightsBuyHold[i] = pd.DataFrame(self.weightsBuyHold[:, :, i].T)
                weightsBuyHold[i].columns = self.assetNames

                outSample[i] = pd.DataFrame(self.outSample[i, :].T)
                outSample[i].columns = ["Out of Sample"]

            weights = pd.concat(weights, keys=gammas, names=["Gamma"])
            weightsBuyHold = pd.concat(weightsBuyHold, keys=gammas, names=["Gamma"])
            outSample = pd.concat(outSample, keys=gammas, names=["Gamma"])

            return weights, weightsBuyHold, outSample

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

    def _init(self, nRisky, period, timeHorizon, riskFreeReturns, riskyReturns):
        self.weights = np.empty((nRisky, period - timeHorizon[-1]))
        self.weightsBuyHold = np.empty(
            (nRisky, period - timeHorizon[-1]))
        self.outSample = np.empty((1, period - timeHorizon[-1]))

        self.riskFreeReturns = riskFreeReturns
        self.riskyReturns = riskyReturns

    def runOutSample(self, params):
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

    def _statisticalSignificanceWRTBenchmark(self, benchmark, nSubsets):
        z = jobsonKorkieZStat(benchmark, self.outSample, nSubsets)
        p = pValue(z)

        return p

# Models that use gamma
class ModelGamma(Model):
    def __init__(self, name):
        super().__init__(name)

    def _init(self, nRisky, period, timeHorizon, riskFreeReturns, riskyReturns, gammas):
        self.weights = np.empty((nRisky, period - timeHorizon[-1], len(gammas)))
        self.weightsBuyHold = np.empty(
            (nRisky, period - timeHorizon[-1], len(gammas)))
        self.outSample = np.empty((len(gammas), period - timeHorizon[-1]))

        self.riskFreeReturns = riskFreeReturns
        self.riskyReturns = riskyReturns

    def runOutSample(self, params):
        currentSubset = params['currentSubset']
        period = params['period']
        gammas = params['gammas']

        filter = filterParams(params, self, "alpha")

        for i in range(0, len(gammas)):

            filter['currentGamma'] = gammas[i]

            alpha = self.alpha(**filter)
            self.weights[:, currentSubset, i] = alpha[:, 0]
            self.outSample[i, currentSubset] = self.outOfSampleReturns(alpha, currentSubset, period)[:, 0]

        if currentSubset == 1:
            self.weightsBuyHold[:, currentSubset,i] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset,i] = self.buyHold(
                self.weights[:, currentSubset - 1, i], currentSubset, period)

    def _statisticalSignificanceWRTBenchmark(self, benchmark, nSubsets, gammas):
        z = list(range(len(gammas)))

        for i in range(0, len(gammas)):
            z[i] = jobsonKorkieZStat(benchmark, self.outSample[i], nSubsets)

        p = pValue(z)

        return p