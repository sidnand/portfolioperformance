# CODE FOR RUNNING THE MODELS USING THE DATA

import numpy as np

from .models import *
from .enum_policy import Policy

from .utils.statistics import *

class System:
    
    def __init__(self, data, T):
        (self.m, self.n) = data.shape

        self.riskFreeReturns = data[:, 0] # risk-free asset column
        self.riskyReturns = data[:, 1:self.n] # risky asset column, includes risk factor
        self.T = T # estimation window length

        self.nRisky = self.n - 1 # number of risky variables

    """
        Computes the sharpe ratios of all the portfolio policies

        returns : Object of floas denoting the sharpe ratio of each portfolio policy
    """
    def getSharpeRatios(self, gamma):
        w = {} # portfolio policy weights
        wBuyHold = {} # portfolio weights before rebalancing
        outSample = {} # out of sample returns

        for i in Policy:
            w[i.value] = np.empty((self.nRisky, self.m - self.T))
            wBuyHold[i.value] = np.empty((self.nRisky, self.m - self.T))
            outSample[i.value] = np.empty((1, self.m - self.T))

        T = len(self.riskyReturns) # time period
        nSubsets = 1 if self.T == T else T - self.T # if M is the same as time period, then we only have 1 subset

        for shift in range(0, nSubsets):

            riskySubset = self.riskyReturns[shift:self.T + shift, :]
            riskFreeSubset = self.riskFreeReturns[shift:self.T + shift]
            subset = np.column_stack((riskFreeSubset, riskySubset))

            nRisky = len(riskySubset)
            
            mu_horz = np.array([np.mean(riskFreeSubset)])
            mu = np.append(mu_horz, np.vstack(riskySubset.mean(axis = 0)))
            
            totalSigma = np.cov(subset.T)
            sigma = (self.T - 1) / (self.T - self.nRisky - 2) * np.cov(riskySubset.T)
            
            sigmaMLE = (self.T - 1) / self.T * np.cov(riskySubset.T)
            invSigmaMLE = np.linalg.inv(sigmaMLE)

            AMLE = np.ones((1, self.n - 1)) @ invSigmaMLE @ np.ones((self.n - 1, 1))

            # 0: 1/N
            alphaTew = ew(self.n)
            w[Policy.EW][:, shift] = alphaTew[:, 0]
            
            # 5: minimum-variance
            alphaMinV = minVar(invSigmaMLE, AMLE, self.n)
            w[Policy.MINIMUM_VAR][:, shift] = alphaMinV[:, 0]

            # 10: minimum-variance shortsell constraints
            minVarCon = minVarShortSellCon(sigmaMLE)
            w[Policy.MINIMUM_VAR_CONSTRAINED][:, shift] = minVarCon[:, 0]

            # 11: minimum-variance generalized constraints
            minVarGCon = jagannathanMa(sigma)
            w[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, shift] = minVarGCon[:, 0]

            # 12 : Kan and Zhou’s (2007) “three-fund” model
            alphaKanZhouEw = kanZhouEw(self.nRisky, self.T, sigma)
            w[Policy.KAN_ZHOU_EW][:, shift] = alphaKanZhouEw[:, 0]

            for i in gamma:
                #  : Mean Variance
                alphaMeanV = meanVar(i, invSigmaMLE, mu)
                wBuyHold[Policy.MEAN_VARIANCE][:, shift]= alphaMeanV[:, 0]

            # buy and hold
            if shift == 0:
                wBuyHold[Policy.EW][:, shift]= alphaTew[:, 0]
                wBuyHold[Policy.MINIMUM_VAR][:, shift]= alphaMinV[:, 0]
                wBuyHold[Policy.MINIMUM_VAR_CONSTRAINED][:, shift] = minVarCon[:, 0]
                wBuyHold[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, shift] = minVarGCon[:, 0]
                wBuyHold[Policy.KAN_ZHOU_EW][:, shift] = alphaKanZhouEw[:, 0]

                wBuyHold[Policy.MEAN_VARIANCE][:, shift] = alphaMeanV[:, 0]
            else:
                wBuyHold[Policy.EW][:, shift] = self.buyHold(w[Policy.EW][:, shift - 1], shift)
                wBuyHold[Policy.MINIMUM_VAR][:, shift] = self.buyHold(w[Policy.MINIMUM_VAR][:, shift - 1], shift)
                wBuyHold[Policy.MINIMUM_VAR_CONSTRAINED][:, shift] = self.buyHold(w[Policy.MINIMUM_VAR_CONSTRAINED][:, shift - 1], shift)
                wBuyHold[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, shift] = self.buyHold(w[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, shift - 1], shift)
                wBuyHold[Policy.KAN_ZHOU_EW][:, shift] = self.buyHold(w[Policy.KAN_ZHOU_EW][:, shift - 1], shift)

                wBuyHold[Policy.MEAN_VARIANCE][:, shift] = self.buyHold(w[Policy.MEAN_VARIANCE][:, shift - 1], shift)

            if (nSubsets > 1):
                # out of sample returns
                outSample[Policy.EW][:, shift] = self.outOfSampleReturns(alphaTew, shift)[:, 0]
                outSample[Policy.MINIMUM_VAR][:, shift] = self.outOfSampleReturns(alphaMinV, shift)[:, 0]
                outSample[Policy.MINIMUM_VAR_CONSTRAINED][:, shift] = self.outOfSampleReturns(minVarCon, shift)[:, 0]
                outSample[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED][:, shift] = self.outOfSampleReturns(minVarGCon, shift)[:, 0]
                outSample[Policy.KAN_ZHOU_EW][:, shift] = self.outOfSampleReturns(alphaKanZhouEw, shift)[:, 0]
                outSample[Policy.MEAN_VARIANCE][:, shift] = self.outOfSampleReturns(alphaMeanV, shift)[:, 0]

        sharpeRatios = {}

        sharpeRatios[Policy.EW] = sharpeRato(outSample[Policy.EW])
        sharpeRatios[Policy.MINIMUM_VAR] = sharpeRato(outSample[Policy.MINIMUM_VAR])
        sharpeRatios[Policy.MINIMUM_VAR_CONSTRAINED] = sharpeRato(outSample[Policy.MINIMUM_VAR_CONSTRAINED])
        sharpeRatios[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED] = sharpeRato(outSample[Policy.MINIMUM_VAR_GENERALIZED_CONSTRAINED])
        sharpeRatios[Policy.KAN_ZHOU_EW] = sharpeRato(outSample[Policy.KAN_ZHOU_EW])
        sharpeRatios[Policy.MEAN_VARIANCE] = sharpeRato(outSample[Policy.MEAN_VARIANCE])

        return sharpeRatios

    """
        Computes a new portfolio weight after a shift

        param w : [n, row - M] array, holds portfolio weights of a specific policy
        param j : integer value, represents current shift position
    """
    def buyHold(self, w, j):

        a = (1 - sum(w)) * (1 + self.riskFreeReturns[self.T + j])
        b = (1 + (self.riskyReturns[self.T + j, :].T + self.riskFreeReturns[self.T + j]))[np.newaxis].T
        trp = a + w[np.newaxis].dot(b)
        
        return ((w * (1 + (self.riskyReturns[self.T + j, :]).T + self.riskFreeReturns[self.T + j])) / trp)

    """
        Computes the out of sample returns

        param w : [n, row - M] array, holds portfolio weights of a specific policy
        param j : integer value, represents current shift position
    """
    def outOfSampleReturns(self, w, j):
        return w.T.dot(self.riskyReturns[self.T + j, :][np.newaxis].T)