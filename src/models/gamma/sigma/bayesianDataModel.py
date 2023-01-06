import numpy as np
import pandas as pd

# from ....modelGammaSigma import ModelGammaSigma
from ....model import Model

from ....utils.filter import filterParams

class BayesianDataModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def _init(self, nRisky, period, timeHorizon, riskFreeReturns, riskyReturns, gammas, omegas):
        self.weights = np.empty(
            (nRisky, period - timeHorizon[-1], len(gammas), len(omegas)))
        self.weightsBuyHold = np.empty(
            (nRisky, period - timeHorizon[-1], len(gammas), len(omegas)))
        self.outSample = np.empty((len(gammas), period - timeHorizon[-1], len(omegas)))

        self.riskFreeReturns = riskFreeReturns
        self.riskyReturns = riskyReturns

    def runOutSample(self, params):
        currentSubset = params['currentSubset']
        period = params['period']
        gammas = params['gammas']
        omegas = params['omegas']

        filter = filterParams(params, self, "alpha")
