function result = dgu_static_buyholdRet(alpha0, returns, M, n, T, nRRA)

%computes the excess return of a buy-hold strategy


clear temp weights wealth xrp

wealth0 = 1;



if length(size(alpha0))<3
    weights= [1-sum(alpha0), alpha0'];
    temp = sum(repmat(weights,length(M+1:T),1).*cumprod(returns),2)';
    wealth = [wealth0, temp];
    xrp = wealth(2:end)./wealth(1:end-1) - returns(:,1)';    %return in excess of RF


else
    for i =1:nRRA
        weights(i,:) = [1-sum(alpha0(:,:,i)), alpha0(:,:,i)'];
        temp(i, :) =sum(repmat(weights(i,:),length(M+1:T),1).*cumprod(returns),2)';
        wealth(i,:) = [wealth0, temp(i,:)];
        xrp(i,:) = wealth(i,2:end)./wealth(i,1:end-1) - returns(:,1)' ;    
        
    end
end

result = xrp;



