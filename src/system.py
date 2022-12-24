# CODE FOR RUNNING THE MODELS USING THE DATA

import numpy as np

from .models import *
from .enum_policy import Policy

from .utils.statistics import *

class System:
    
    def __init__(self, data, timeHorizon):
        (self.m, self.n) = data.shape

        self.riskFreeReturns = data[:, 0] # risk-free asset column
        self.riskyReturns = data[:, 1:self.n] # risky asset column, includes risk factor
        self.timeHorizon = timeHorizon # estimation window length

        self.nRisky = self.n - 1 # number of risky variables

    def getSharpeRatios(self, gamma):
        w = {} # portfolio policy weights
        wBuyHold = {} # portfolio weights before rebalancing
        outSample = {} # out of sample returns

        for i in Policy:
            w[i.value] = np.empty((self.nRisky, self.m - self.timeHorizon[-1]))
            wBuyHold[i.value] = np.empty((self.nRisky, self.m - self.timeHorizon[-1]))
            outSample[i.value] = np.empty((1, self.m - self.timeHorizon[-1]))

        T = len(self.riskyReturns) # time period
        upperM = self.timeHorizon[-1] # upper bound of time horizon

        for k in self.timeHorizon:
            M = k # time horizon
            shift = upperM - M # shift in time horizon
            M = M + shift # update time horizon
            
            nSubsets = 1 if M == T else T - M # if M is the same as time period, then we only have 1 subset

            for j in range(0, nSubsets):

                riskySubset = self.riskyReturns[j+shift:M+j-1, :]
                riskFreeSubset = self.riskFreeReturns[j+shift:M+j-1]
                subset = np.column_stack((riskFreeSubset, riskySubset))

                nRisky = len(riskySubset)
                
                mu_horz = np.array([np.mean(riskFreeSubset)])
                mu = np.append(mu_horz, np.vstack(riskySubset.mean(axis = 0)))
                
                totalSigma = np.cov(subset.T)
                sigma = (M - 1) / (M - self.nRisky - 2) * np.cov(riskySubset.T)
                
                sigmaMLE = (M - 1) / M * np.cov(riskySubset.T)
                invSigmaMLE = np.linalg.inv(sigmaMLE)

                AMLE = np.ones((1, self.nRisky)) @ invSigmaMLE @ np.ones((self.nRisky, 1))

                # 0: 1/N
                alphaTew = ew(self.n)
                w[Policy.EW][:, j] = alphaTew[:, 0]
                
                # 5: minimum-variance
                alphaMinV = minVar(invSigmaMLE, AMLE, self.n)
                w[Policy.MINIMUM_VAR][:, j] = alphaMinV[:, 0]

                # 10: minimum-variance shortsell constraints
                minVarCon = minVarShortSellCon(sigmaMLE)
                w[Policy.MINIMUM_VAR_CONSTRAINED][:, j] = minVarCon[:, 0]

                # 11: minimum-variance generalized constraints
                minVarGCon = jagannathanMa(sigma)
                w[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, j] = minVarGCon[:, 0]

                # 12 : Kan and Zhou’s (2007) “three-fund” model
                alphaKanZhouEw = kanZhouEw(self.nRisky, M, sigma)
                w[Policy.KAN_ZHOU_EW][:, j] = alphaKanZhouEw[:, 0]

                for i in gamma:
                    #  : Mean Variance
                    alphaMeanV = meanVar(i, invSigmaMLE, mu)
                    wBuyHold[Policy.MEAN_VARIANCE][:, j]= alphaMeanV[:, 0]

                # buy and hold
                if j == 0:
                    wBuyHold[Policy.EW][:, j]= alphaTew[:, 0]
                    wBuyHold[Policy.MINIMUM_VAR][:, j]= alphaMinV[:, 0]
                    wBuyHold[Policy.MINIMUM_VAR_CONSTRAINED][:, j] = minVarCon[:, 0]
                    wBuyHold[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, j] = minVarGCon[:, 0]
                    wBuyHold[Policy.KAN_ZHOU_EW][:, j] = alphaKanZhouEw[:, 0]

                    wBuyHold[Policy.MEAN_VARIANCE][:, j] = alphaMeanV[:, 0]
                else:
                    wBuyHold[Policy.EW][:, j] = self.buyHold(w[Policy.EW][:, j - 1], j, M)
                    wBuyHold[Policy.MINIMUM_VAR][:, j] = self.buyHold(w[Policy.MINIMUM_VAR][:, j - 1], j, M)
                    wBuyHold[Policy.MINIMUM_VAR_CONSTRAINED][:, j] = self.buyHold(w[Policy.MINIMUM_VAR_CONSTRAINED][:, j - 1], j, M)
                    wBuyHold[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, j] = self.buyHold(w[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, j - 1], j, M)
                    wBuyHold[Policy.KAN_ZHOU_EW][:, j] = self.buyHold(w[Policy.KAN_ZHOU_EW][:, j - 1], j, M)

                    wBuyHold[Policy.MEAN_VARIANCE][:, j] = self.buyHold(w[Policy.MEAN_VARIANCE][:, j - 1], j, M)

                if (nSubsets > 1):
                    # out of sample returns
                    outSample[Policy.EW][:, j] = self.outOfSampleReturns(alphaTew, j, M)[:, 0]
                    outSample[Policy.MINIMUM_VAR][:, j] = self.outOfSampleReturns(alphaMinV, j, M)[:, 0]
                    outSample[Policy.MINIMUM_VAR_CONSTRAINED][:, j] = self.outOfSampleReturns(minVarCon, j, M)[:, 0]
                    outSample[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, j] = self.outOfSampleReturns(minVarGCon, j, M)[:, 0]
                    outSample[Policy.KAN_ZHOU_EW][:, j] = self.outOfSampleReturns(alphaKanZhouEw, j, M)[:, 0]
                    outSample[Policy.MEAN_VARIANCE][:, j] = self.outOfSampleReturns(alphaMeanV, j, M)[:, 0]

        statistics = {}

        benchmark = outSample[Policy.EW]

        statistics[Policy.EW] = getStats(benchmark, benchmark, nSubsets)
        statistics[Policy.MINIMUM_VAR] = getStats(benchmark, outSample[Policy.MINIMUM_VAR], nSubsets)
        statistics[Policy.MINIMUM_VAR_CONSTRAINED] = getStats(benchmark, outSample[Policy.MINIMUM_VAR_CONSTRAINED], nSubsets)
        statistics[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED] = getStats(benchmark, outSample[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED], nSubsets)
        statistics[Policy.KAN_ZHOU_EW] = getStats(benchmark, outSample[Policy.KAN_ZHOU_EW], nSubsets)
        # statistics[Policy.MEAN_VARIANCE] = getStats(outSample[Policy.MEAN_VARIANCE])

        return statistics

    def buyHold(self, w, j, M):
        a = (1 - sum(w)) * (1 + self.riskFreeReturns[M + j])
        b = (1 + (self.riskyReturns[M + j, :].T + self.riskFreeReturns[M + j]))[np.newaxis].T
        trp = a + w[np.newaxis].dot(b)
        
        return ((w * (1 + (self.riskyReturns[M + j, :]).T + self.riskFreeReturns[M + j])) / trp)

    def outOfSampleReturns(self, w, j, M):
        return w.T.dot(self.riskyReturns[M + j, :][np.newaxis].T)