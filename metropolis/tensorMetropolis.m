outSteps = 100000;
inSteps = 100;
N = 10;
f = exp(1);
b = 0.92;
nBins = 10;
width = 6;
binLambdaMin = 1;
binLambdaMax = 8;
[logWeights, nc, f, histogram, accepted, rejected] = metropolisEigenValues(outSteps, inSteps, N, f, b, nBins, width, binLambdaMin, binLambdaMax);
writematrix(logWeights, "./MatlabWeights/Weights.txt");
writematrix(histogram, "./MatlabWeights/Histogram.txt");

function count = d(i, j, k)
    if i == j && j==k
        count = 1;
    elseif i == j || j == k || i == k
        count = 3;
    else 
        count = 6;
    end
end

function prob = pijk(xijk, i, j, k)
    prob = exp(-xijk^2 / 2 / d(i, j, k));
end

function bin = computeBin(lambdaC, binLambdaMin, binLambdaMax, nBins)
    if lambdaC > binLambdaMax
        bin = nBins;
    elseif lambdaC < binLambdaMin
        bin = 1;
    else
        bin = round((lambdaC - binLambdaMin) / (binLambdaMax - binLambdaMin) * (nBins-1)) + 1;
    end
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

function A = getSymmetricTensor(N)
    A = symtensor(@randn, 3, N);

    for i=1:N
        for j=1:i
            for k=1:j
                if i==j && j==k
                    continue
                elseif i==j || i==k || j==k
                    A(i, j, k) = A(i, j, k) / 3;
                else
                    A(i, j, k) = A(i, j, k) / 6;
                end
            end
        end
    end 
end

function [logWeights, nc, f, histogram, accepted, rejected] = metropolisEigenValues(outSteps, inSteps, N, f, b, nBins, width, binLambdaMin, binLambdaMax)
    logWeights = zeros(nBins, 1);
    histogram = zeros(nBins, 1);
    nc = 0;

    A = getSymmetricTensor(N);

    [lambda, eigVectors] = zeig(double(full(A)));

    % Filter out negative lambda since they are just duplicates
    eigVectors = eigVectors(:, lambda > 0);
    lambda = lambda(lambda > 0);   
    lambdaMax = max(lambda);

    accepted = 0;
    rejected = 0;
    for outerIdx=1:outSteps
        for innerIdx=1:inSteps
            indices = randi([1, N], 3);
            i = indices(1);
            j = indices(2);
            k = indices(3);
            proposedJump = randn() * width / 6;
            
            Anew = A;
            Anew(i,j,k) = Anew(i, j, k) + proposedJump;
            
            [lambdaNew, eigVectorsNew] = getEigenvalues(eigVectors, Anew);
            lambdaMaxNew = max(lambdaNew);

            binLambdaNew = computeBin(lambdaMaxNew, binLambdaMin, binLambdaMax, nBins);
            binLambda = computeBin(lambdaMax, binLambdaMin, binLambdaMax, nBins);
            monteCarloProbability = pijk(Anew(i, j, k), i, j, k) / pijk(A(i, j, k), i, j, k) * exp(logWeights(binLambdaNew) - logWeights(binLambda));

            if isnan(monteCarloProbability)
                throw(MException("Nan"));
            end

            if rand < monteCarloProbability
                lambdaMax = lambdaMaxNew;
                eigVectors = eigVectorsNew;
                A = Anew;
                accepted = accepted + 1;
            else
                rejected = rejected + 1;
            end

            binLambda = computeBin(lambdaMax, binLambdaMin, binLambdaMax, nBins);
            logWeights(binLambda) = logWeights(binLambda) - log(f);
            histogram(binLambda) = histogram(binLambda) + 1;
        end
        display([accepted / (accepted + rejected) * 100, lambdaMax]);
        choppedHistogram = histogram(2:end-1);
        % nonZeroChopped = choppedHistogram(choppedHistogram ~= 0);

        if min(choppedHistogram) > mean(choppedHistogram) * b
            f = sqrt(f);
            nc = nc + 1; 
            writematrix(logWeights, sprintf("./MatlabWeights/Weights_%d_%d.txt", nc, N));
            writematrix(histogram, sprintf("./MatlabWeights/Histogram_%d_%d.txt", nc, N));
            histogram = histogram * 0;
            
        end
        
    end
end