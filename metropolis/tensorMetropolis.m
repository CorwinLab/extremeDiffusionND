function prob = pij(xij)
    prob = exp(-xij^2 / 2);
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
    
    parfor eigIdx=1:numEigVectors
        [powerLambda, powerV] = eig_sshopm(tensor(A), 'Start', prevEigVectors(:, eigIdx));
        eigValues(eigIdx) = powerLambda;
        eigVectors(:, eigIdx) = powerV; 
    end
end

function [logWeights, nc, f, histogram, accepted, rejected] = metropolisEigenValues(outSteps, inSteps, N, f, b, nBins, width, binLambdaMin, binLambdaMax)
    logWeights = zeros(nBins, 1);
    histogram = zeros(nBins, 1);
    nc = 0;

    A = symtensor(@randn, 3, N);
    A = double(full(A));

    [lambda, eigVectors] = zeig(A);

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
            % Enforce all the symmetries of the tensor
            Anew(i, j, k) = Anew(i, j, k) + proposedJump;
            Anew(i, k, j) = Anew(i, k, j) + proposedJump;
            Anew(k, i, j) = Anew(k, i, j) + proposedJump;
            Anew(k, j, i) = Anew(k, j, i) + proposedJump;
            Anew(j, i, k) = Anew(j, i, k) + proposedJump;
            Anew(j, k, i) = Anew(j, k, i) + proposedJump;
            
            [lambdaNew, eigVectorsNew] = getEigenvalues(eigVectors, tensor(Anew));
            lambdaMaxNew = max(lambdaNew);

            binLambdaNew = computeBin(lambdaMaxNew, binLambdaMin, binLambdaMax, nBins);
            binLambda = computeBin(lambdaMax, binLambdaMin, binLambdaMax, nBins);
            monteCarloProbability = pij(Anew(i, j, k)) / pij(A(i, j, k)) * exp(logWeights(binLambdaNew) - logWeights(binLambda));

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
        display([accepted, rejected]);
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

outSteps = 100000;
inSteps = 100;
N = 10;
f = exp(1);
b = 0.92;
nBins = 30;
width = 6;
binLambdaMin = 4.5;
binLambdaMax = 16;
[logWeights, nc, f, histogram, accepted, rejected] = metropolisEigenValues(outSteps, inSteps, N, f, b, nBins, width, binLambdaMin, binLambdaMax);
writematrix(logWeights, "./MatlabWeights/Weights.txt");
writematrix(histogram, "./MatlabWeights/Histogram.txt");