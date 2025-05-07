warning("off", 'all');

function [c,ceq] = unitdisk(x)
    c = [];
    ceq = norm(x) - 1;
end

rng("shuffle")

A = [];
b = [];
Aeq = []; 
beq = [];
lb = [];
ub = []; 
nonlcon = @unitdisk;
N = 3;

X = randn(N, N, N, 2);
Atensor = tensor(X(:, :, :, 1));
fun = @(x) ttv(Atensor, {x, x, x}, [1 2 3]);

symmetrizedTensor = (permute(X, [1 2 3 4]) + permute(X, [3 1 2 4]) + permute(X, [2 3 1 4]));
symmetrizedTensor = symmetrizedTensor(:, :, :, 1) / 3;
lambda = zeig(symmetrizedTensor);
display(lambda);

fullySymmetrizedTensor = 1/6 * (permute(X, [1 2 3 4]) + permute(X, [1 3 2 4]) + permute(X, [2 1 3 4]) + permute(X, [2 3 1 4]) + permute(X, [3 2 1 4]) + permute(X, [3 1 2 4]));
lambda = zeig(fullySymmetrizedTensor(:, :, :, 1), 'symmetric');
display(lambda);

energies =[]; 
for i=1:100
    x0 = rand(N, 1);
    x0 = x0 / norm(x0);
    
    options = optimoptions('fmincon','Algorithm','interior-point', 'Display', 'off');
    [x, fval, exitflag] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
    if exitflag ~=1
        continue
    end
    energies(end+1) = fval;
end

energies = uniquetol(energies, 0.01);
display(energies);