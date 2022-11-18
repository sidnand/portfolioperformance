function xrp = dgu_static_outSample(alpha, rRisky,M,j)

%Compute out of sample excess return

%Excess returns:
% [alpha'(grossRet) + (1-sum(alpha))*RF] - RF = alpha'(exRet)
% rRisky represents excess returns
xrp = alpha'*(rRisky(M+j,:))';
   