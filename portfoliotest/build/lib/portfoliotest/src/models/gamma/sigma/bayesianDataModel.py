import numpy as np
import pandas as pd

# from ....modelGammaSigma import ModelGammaSigma
from ....model import Model

from ....utils.filter import filterParams


class BayesianDataModel(Model):
    def __init__(self) -> None:
        super().__init__()

    def _init(self, nRisky, nPoints, timeHorizon, riskFreeReturns, riskyReturns, gammas, omegas):
        self.weights = np.empty(
            (nRisky, nPoints - timeHorizon[-1], len(gammas), len(omegas)))
        self.weightsBuyHold = np.empty(
            (nRisky, nPoints - timeHorizon[-1], len(gammas), len(omegas)))
        self.outSample = np.empty(
            (len(gammas), nPoints - timeHorizon[-1], len(omegas)))

        self.riskFreeReturns = riskFreeReturns
        self.riskyReturns = riskyReturns

    def runOutSample(self, params):
        currentSubset = params['currentSubset']
        nPoints = params['nPoints']
        gammas = params['gammas']
        omegas = params['omegas']
        withoutRiskFactorReturns = params['withoutRiskFactorReturns']
        windowType = params['windowType']

        filter = filterParams(params, self, "preAlpha")

    def preAlpha():
        rfactorSubset = None

        if windowType == "Rolling":
            rfactorSubset = withoutRiskFactorReturns[currentSubset +
                                     shift:nPoints + currentSubset - 1, :]
        else:
            rfactorSubset = withoutRiskFactorReturns[1 +
                                                     shift:nPoints + currentSubset - 1, :]

        muFactor = np.mean(rfactorSubset, axis=0)
        sigmaFactor = np.cov(rfactorSubset, rowvar=False)
        sharpe = ((muFactor * np.linalg.inv(sigmaFactor)) @ muFactor.T) ** (1/2)

    def alpha():
        pass
