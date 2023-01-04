import numpy as np

from .model import Model
from src.utils.filter import filterParams
from src.utils.statistics import *

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
