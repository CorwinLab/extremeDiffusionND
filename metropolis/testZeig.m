function count = d(indices)
    i = indices(1);
    j = indices(2);
    k = indices(3);
   
    if i == j && j==k
        count = 1;
    elseif i == j || j == k || i == k
        count = 3;
    else 
        count = 6;
    end
end

function prob = pijConstantVariance(xijk, indices)
    prob = exp(-xijk^2 / 2);
end

function prob = pijkCyclicVariance(xijk, indices)
    prob = exp(-xijk^2 / 2 * d(indices));
end

function A = getCyclicTensor(N)
    A = symtensor(@randn, 3, N);

    for i=1:N
        for j=1:i
            for k=1:j
                if i==j && j==k
                    continue
                elseif i==j || i==k || j==k
                    A(i, j, k) = A(i, j, k) / sqrt(3);
                else
                    A(i, j, k) = A(i, j, k) / sqrt(6);
                end
            end
        end
    end 
end

function A = getConstantVarianceTensor(N)
    A = symtensor(@randn, 3, N);
end

function [eigValues, eigVectors] = getEigenvalues(prevEigVectors, A)
    % Let's say that prevEigVectors is size (N, numEigVectors)
    % Tensor is a "tensor" object
    s = size(prevEigVectors);
    numEigVectors = s(2);

    eigValues = zeros(numEigVectors, 1);
    eigVectors = zeros(size(prevEigVectors));

    for eigIdx=1:numEigVectors
        [powerLambda, powerV] = eig_sshopm(tensor(A), 'Start', prevEigVectors(:, eigIdx));
        eigValues(eigIdx) = powerLambda;
        eigVectors(:, eigIdx) = powerV; 
    end
end

ensemble = 'Cyclic';
N = 3;
width = 1/sqrt(6);

if strcmp(ensemble, 'Cyclic')
    A = getCyclicTensor(N);
    pijk = @pijkCyclicVariance;
elseif strcmp(ensemble, 'ConstantVariance')
    A = getConstantVarianceTensor(N);
    pijk = @pijConstantVariance;
else
    ME = MException("Ensemble is not correct");
    throw(ME);
end

s = rng;
[lambda, eigVectors] = zeig(double(full(A)));
rng(s);

% Filter out negative lambda since they are just duplicates
eigVectors = eigVectors(:, lambda > 0);
lambda = lambda(lambda > 0);   
lambdaMax = max(lambda);

num_samples = 1000;

correctEigValues = zeros(num_samples, 1);
powerEigValues = zeros(num_samples, 1);

for i=1:num_samples
    indices = randi([1, N], 1, 3);
    proposedJump = randn() * width;

    Anew = A;
    Anew(indices) = Anew(indices) + proposedJump;
    monteCarloProbability = pijk(Anew(indices), indices) / pijk(A(indices), indices);
    
    [lambdaNew, eigVectorsNew] = getEigenvalues(eigVectors, Anew);
    lambdaMaxNew = max(lambdaNew);
    powerEigValues(i) = lambdaMaxNew;

    lambda = zeig(double(full(Anew)));
    correctEigValues(i) = max(lambda);

    if rand < monteCarloProbability
        A = Anew;
        lambdaMax = lambdaMaxNew;
        eigVectors = eigVectorsNew;
    end
end

numCorrect = abs(correctEigValues - powerEigValues) < 1e-8;
display(sum(numCorrect) / num_samples * 100);


