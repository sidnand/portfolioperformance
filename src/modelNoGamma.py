import numpy as np

from .model import Model
from src.utils.filter import filterParams
from src.utils.statistics import *

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
