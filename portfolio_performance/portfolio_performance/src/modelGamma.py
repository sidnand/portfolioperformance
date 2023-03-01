import numpy as np
import pandas as pd

from .model import Model
from .utils.filter import filterParams
from .utils.statistics import *

# Models that use gamma
class ModelGamma(Model):
    def __init__(self, name):
        super().__init__(name)

    def _init(self, nRisky, period, timeHorizon, riskFreeReturns, riskyReturns, gammas):
        self.weights = np.empty(
            (nRisky, period - timeHorizon[-1], len(gammas)))
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
            self.outSample[i, currentSubset] = self.outOfSampleReturns(
                alpha, currentSubset, period)[:, 0]

        if currentSubset == 1:
            self.weightsBuyHold[:, currentSubset, i] = alpha[:, 0]
        else:
            self.weightsBuyHold[:, currentSubset, i] = self.buyHold(
                self.weights[:, currentSubset - 1, i], currentSubset, period)

    def _statisticalSignificanceWRTBenchmark(self, benchmark, nSubsets, gammas):
        z = list(range(len(gammas)))

        for i in range(0, len(gammas)):
            z[i] = jobsonKorkieZStat(benchmark, self.outSample[i], nSubsets)

        p = pValue(z)

        return p

    def toDataFrame(self, **kwargs):
        gammas = kwargs['gammas']

        weights = [None] * len(gammas)
        weightsBuyHold = [None] * len(gammas)
        outSample = [None] * len(gammas)

        for i in range(0, len(gammas)):
            weights[i] = pd.DataFrame(self.weights[:, :, i].T)
            weights[i].columns = self.assetNames

            weightsBuyHold[i] = pd.DataFrame(
                self.weightsBuyHold[:, :, i].T)
            weightsBuyHold[i].columns = self.assetNames

            outSample[i] = pd.DataFrame(self.outSample[i, :].T)
            outSample[i].columns = ["Out of Sample"]

        weights = pd.concat(weights, keys=gammas, names=["Gamma"])
        weightsBuyHold = pd.concat(
            weightsBuyHold, keys=gammas, names=["Gamma"])
        outSample = pd.concat(outSample, keys=gammas, names=["Gamma"])

        return weights, weightsBuyHold, outSample

    def statisticalSignificanceSR0(self, sr, **kwargs):
        gammas = kwargs['gammas']

        z = list(range(len(gammas)))

        for i in range(0, len(gammas)):
            z[i] = zSharpeRatio0(self.outSample[i], sr)

        p = pValue(z)

        return p
