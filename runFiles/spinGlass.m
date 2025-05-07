warning("off", 'all');

function [c,ceq] = unitdisk(x)
    c = [];
    ceq = norm(x) - 1;
end

A = [];
b = [];
Aeq = []; 
beq = [];
lb = [];
ub = []; 
nonlcon = @unitdisk;
N = 3;
numSamples = 10000;
minEnergies = Inf(numSamples, 1);

for j=1:numSamples
    X = randn(N, N, N);
    Atensor = tensor(X);
    fun = @(x) ttv(Atensor, {x, x, x}, [1 2 3]);
    disp(j);

    for i=1:10
        x0 = rand(N, 1);
        x0 = x0 / norm(x0);
        options = optimoptions('fmincon','Algorithm','interior-point', 'Display', 'off');
        [x, fval, exitflag] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
        if exitflag ~= 1
            continue
        end
        if fval < minEnergies(j)
            minEnergies(j) = fval;
        end
    end
end
writematrix(minEnergies, "MinimumEnergies.txt");