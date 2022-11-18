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

        %Collect mean and variances
        forecastMuMLE(j,:) = mu';
        forecastSigmaMLE(:,:,j) = Sigma;
        detSigmaMLE(j) = det(Sigma);
        condSigmaMLE(j) = cond(Sigma);
        
        %Bayes-Stein moments (From Jorion, JFQA 1896, pp. 285-286)
        Y = mu(2:end);
        SigmaHat = (nPoints-1)/(nPoints-nRisky -2)*cov(rRisky_subset);
        invSigmaHat = inv(SigmaHat);
        Ahat = ones(1,nRisky)*invSigmaHat*ones(nRisky,1);
        Y0 = (ones(1,nRisky)*invSigmaHat*Y)/Ahat;
        w = (nRisky+2)/((nRisky+2) + (Y-Y0)'*nPoints*invSigmaHat*(Y-Y0));
        lambda = (nRisky+2)/((Y-Y0)'*invSigmaHat*(Y-Y0));
        muBS = [mean(rRiskfree_subset);(1-w)*Y + w*Y0];
        SigmaBS = SigmaHat*(1+1/(nPoints+lambda)) + lambda/(nPoints*(nPoints+1+lambda))*ones(nRisky,1)*ones(1,nRisky)/Ahat;
        invSigmaBS = inv(SigmaBS);
        totalSigmaBS = (nPoints-1)/(nPoints-n-2)*totalSigma;
        bsPhi(j) = w;

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

        % Loop for gamma
        for i=1:nRRA

            gam = gamma_values(i);
            
            %MEAN-VARIANCE
            alphaMV   = 1/gam*invSigma*(mu(2:end));
            pf.MV(:,j,i) = alphaMV;
            xrp.mv(i,j)   = dgu_static_outSample(alphaMV, rRisky,M,j);


            %Buy and Hold
            if j==1
                pfBuyHold.MV(:,j,i) = alphaMV;
            else
                pfBuyHold.MV(:,j,i) = dgu_static_buyhold(pf.MV(:,j-1,i),j,M, rRiskfree, rRisky);
            end

        end

    end

    meanPhi = mean(bsPhi);
    stdPhi = std(bsPhi);
    
    if nSubsets>1
        
        %---------------------------------------------
        % 4. OUT-OF-SAMPLE PERFORMANCE
        %---------------------------------------------
        %BUY-AND-HOLD OUT OF SAMPLE RETURNS
        
        returns = [1+rRiskfree(M+1:end,:),1+rRisky(M+1:end,:)+repmat(rRiskfree(M+1:end,:),1,n-1)];
        
        xrp.ewbh      = dgu_static_buyholdRet(pfBuyHold.ew(:,1),returns, M, n, T,nRRA);
        xrp.minvbh    = dgu_static_buyholdRet(pfBuyHold.MinVar(:,1),returns, M, n, T,nRRA);
        xrp.minvbhCon = dgu_static_buyholdRet(pfBuyHold.MinVarCon(:,1),returns, M, n, T,nRRA);
        xrp.kz1Nbh    = dgu_static_buyholdRet(pfBuyHold.KZ1N(:,1),returns, M, n, T,nRRA);
        xrp.jmbh      = dgu_static_buyholdRet(pfBuyHold.JM(:,1),returns, M, n, T,nRRA);
        
        xrp.mvbh      = dgu_static_buyholdRet(pfBuyHold.MV(:,1,:),returns, M, n, T,nRRA);
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
[mn.mv,       stdev.mv,       sr.mv]        = dgu_static_SharpeRatios(xrp.mv);

disp(sr)