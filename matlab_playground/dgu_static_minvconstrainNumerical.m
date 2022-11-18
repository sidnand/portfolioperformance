function result = dgu_static_minvconstrainNumerical(Sigma)

[n,n] = size(Sigma);

Aeq = ones(1,n);
beq =1;
lb = zeros(1,n);
ub = ones(1,n);

H = Sigma;
f = zeros(n,1);

options = optimset('Display','off');
x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[], options);

result = x;


