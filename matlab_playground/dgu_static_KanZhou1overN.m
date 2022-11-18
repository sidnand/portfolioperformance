function alpha = dgu_static_KanZhou1overN(N, T, Sigma) 

% Implement three fund portfolio a la Kan and Zhou with 1/N and minimum
% variance as the two risky portfolios

invSigma = inv(Sigma);
esige    = ones(1,N)*   Sigma*ones(N,1);
einvsige = ones(1,N)*invSigma*ones(N,1);

k        = (T^2*(T-2))/((T-N-1)*(T-N-2)*(T-N-4));

d        = ((T-N-2)*esige*einvsige - N^2*T)/(N^2*(T-N-2)*k*einvsige - 2*T*N^2*einvsige + (T-N-2)*einvsige^2*esige);
c        = 1-d*einvsige;

alpha    = c*1/N*ones(N,1) + d*invSigma*ones(N,1);