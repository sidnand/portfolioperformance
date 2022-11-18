function result = dgu_static_JagannathanMa(Sigma)

[n,n] = size(Sigma);

Aeq = ones(1,n);
beq =1;
lb = ones(1,n)/(2*n);    %Generalize the minimum variance constrained
ub = ones(1,n);

H = Sigma;
f = zeros(n,1);

options = optimset('Display','off');
x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[], options);

result = x;
