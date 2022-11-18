function result = dgu_static_buyhold(alpha, j,M, rRiskfree, rRisky)

%Computes the portfolio weights before rebalancing

%Use the non-log version of the return to get total returns
% ret = alpha1*(1+R1)+alpha2*(1+R2) + alpha3*(1+R3),
% 1+R1 = exp(r1), 1+R2 = exp(xr2+r1), 1+R3 = exp(xr3+r1)

trp = (1-sum(alpha))*(1+rRiskfree(M+j)) + alpha'*(1+(rRisky(M+j,:)' +rRiskfree(M+j)));

%Buy-Hold portfolio weights
result = (( alpha.*(1+(rRisky(M+j,:)' +rRiskfree(M+j))))/trp);
