import numpy as np

class Environment():
    def __init__(self, data, timeHorizon, gamma) -> None:
        (self.m, self.n) = data.shape

        self.riskFreeReturns = data[:, 0]  # risk-free asset column
        # risky asset column, includes risk factor
        self.riskyReturns = data[:, 1:self.n]
        self.timeHorizon = timeHorizon  # estimation window length
        self.gamma = gamma

        self.nRisky = self.n - 1  # number of risky variables
        self.T = len(self.riskyReturns)  # time period
        self.upperM = self.timeHorizon[-1]  # upper bound of time horizon