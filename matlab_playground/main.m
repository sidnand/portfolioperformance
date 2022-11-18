warning off

% NoiseLevel = 20;

%---------------------
% 1. PARAMETERS
%---------------------
gamma_values = [1, 2, 3, 4, 5, 10];
nRRA = length(gamma_values);

WindowLengthMonthly = 120;

cd ../data
DataFile = sprintf('SPSectors.txt');
LOG = 0;
RISKFACTOR = 1;    %1= only mkt as a factor
Window = WindowLengthMonthly;
row = 1;   %starting point is 1950-01-28

z=dlmread(DataFile,'\t',row,0);

cd ..
cd matlab_playground

rRiskfree = z(:,2);

rRisky = z(:,3:end);
rFactor = z(:,end-RISKFACTOR+1:end);
n = length(rRiskfree(1,:))+length(rRisky(1,:));

upperM = Window(end);

for k = 1:length(Window)
    
    T = length(rRisky);
    M = Window(k);

    nPoints = M;
    nRisky = n-1;
    
    shift = upperM-M;
    M=M+shift;
    
    if M==T
        nSubsets =1;
    else
        nSubsets = T-M;
    end
    
    %~~~~~~~~~~~~~~~~
    % 3a. ESTIMATION
    %~~~~~~~~~~~~~~~~
    for j=1:nSubsets

        rRisky_subset = rRisky(j+shift:M+j-1,:);
        rRiskfree_subset = rRiskfree(j+shift:M+j-1,:);

        mu = [mean(rRiskfree_subset); mean(rRisky_subset)'];
        totalSigma = cov([rRiskfree_subset,rRisky_subset]);
        
        %Unbiased estimator of sigmainverse
        Sigma = (nPoints-1)/(nPoints-nRisky -2)*cov(rRisky_subset);
        
        %MLE Estimator of Sigma (MINV, MINV-C and Kan Zhou)
        SigmaMLE = (nPoints-1)/(nPoints)*cov(rRisky_subset);
        invSigmaMLE= inv(SigmaMLE);
        AMLE = ones(1,nRisky)*invSigmaMLE*ones(nRisky,1);
        
        invSigma = inv(Sigma);
        A = ones(1,nRisky)*invSigma*ones(nRisky,1);

        %Equal weight policy
        % MODEL 0
        alphaTew     = 1/n*ones(n-1,1);
        pf.ew(:,j) = alphaTew;
        
        %Minimum-Variance
        % MODEL 5
        alphaMinVar  = 1/AMLE*invSigmaMLE*ones(n-1,1);
        pf.MinVar(:,j) = alphaMinVar;
        
        %MinVar-Constrained
        % MODEL 10
        alphaMinVarCon    = dgu_static_minvconstrainNumerical(SigmaMLE);
        pf.MinVarCon(:,j) = alphaMinVarCon;
        
        %Jagannathan-Ma
        % MODEL 11
        alphaJM    = dgu_static_JagannathanMa(Sigma);
        pf.JM(:,j) = alphaJM;
        
        %Kan-Zhou 1 over N
        alphaKZ1N = dgu_static_KanZhou1overN(nRisky, M, Sigma);
        pf.KZ1N(:,j) = alphaKZ1N;

        %Buy and Hold
        if j==1
            pfBuyHold.ew(:,j) = alphaTew;
            pfBuyHold.MinVar(:,j) = alphaMinVar;
            pfBuyHold.MinVarCon(:,j) = alphaMinVarCon;
            pfBuyHold.KZ1N(:,j) = alphaKZ1N;
            pfBuyHold.JM(:,j) = alphaJM;
            
        else
            pfBuyHold.ew(:,j)        = dgu_static_buyhold(pf.ew(:,j-1),j,M, rRiskfree, rRisky);
            pfBuyHold.MinVar(:,j)    = dgu_static_buyhold(pf.MinVar(:,j-1),j,M, rRiskfree, rRisky);
            pfBuyHold.MinVarCon(:,j) = dgu_static_buyhold(pf.MinVarCon(:,j-1),j,M, rRiskfree, rRisky);
            pfBuyHold.KZ1N(:,j)      = dgu_static_buyhold(pf.KZ1N(:,j-1),j,M, rRiskfree, rRisky);
            pfBuyHold.JM(:,j)        = dgu_static_buyhold(pf.JM(:,j-1),j,M, rRiskfree, rRisky);
        end

        if nSubsets>1
            xrp.ew(:,j)      = dgu_static_outSample(alphaTew, rRisky, M,j);
            xrp.minv(:,j)    = dgu_static_outSample(alphaMinVar, rRisky, M,j);
            xrp.minvCon(:,j) = dgu_static_outSample(alphaMinVarCon, rRisky, M,j);
            xrp.kz1N(:,j)    = dgu_static_outSample(alphaKZ1N, rRisky, M,j);
            xrp.jm(:,j)      = dgu_static_outSample(alphaJM, rRisky, M,j);
        end

    end

end

%~~~~~~~~~~~~~~~~~~~
%4a. SHARPE RATIOS
%~~~~~~~~~~~~~~~~~~~%
[mn.ew,       stdev.ew,       sr.ew]        = dgu_static_SharpeRatios(xrp.ew);
[mn.minv,     stdev.minv,     sr.minv]      = dgu_static_SharpeRatios(xrp.minv);
[mn.minvCon,  stdev.minvCon,  sr.minvCon]   = dgu_static_SharpeRatios(xrp.minvCon);
[mn.kz1N,     stdev.kz1N,     sr.kz1N]      = dgu_static_SharpeRatios(xrp.kz1N);
[mn.jm,       stdev.jm,       sr.jm]        = dgu_static_SharpeRatios(xrp.jm);

disp(sr)