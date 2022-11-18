function [meanRet, stdRet, srRet] = dgu_static_SharpeRatios(xrp)

%Computes Sharpe ratios

meanRet = mean(xrp');
stdRet = std(xrp');

for i=1:length(meanRet)

    if abs(meanRet(i))>1e-16
        srRet(i) = meanRet(i)./stdRet(i);
    else
        srRet(i) = NaN;
    end
end
