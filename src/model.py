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

    def toDataFrame(self, **kwargs):
        raise NotImplementedError("Model does not implement toDataFrame method")

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

    def statisticalSignificanceSR0(self, sr, **kwargs):
        raise NotImplementedError("Model does not implement statisticalSignificanceSR0 method")

    def buyHold(self, weights, currentSubset, period):
        a = (1 - sum(weights)) * (1 + self.riskFreeReturns[period + currentSubset])
        b = (1 + (self.riskyReturns[period + currentSubset, :].T +
             self.riskFreeReturns[period + currentSubset]))[np.newaxis].T
        trp = a + weights[np.newaxis].dot(b)

        return ((weights * (1 + (self.riskyReturns[period + currentSubset, :]).T + self.riskFreeReturns[period + currentSubset])) / trp)

    def outOfSampleReturns(self, weights, currentSubset, period):
        return weights.T.dot(self.riskyReturns[period + currentSubset, :][np.newaxis].T)