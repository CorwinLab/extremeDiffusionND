rng('shuffle');

outSteps = 10000;
inSteps = 100;
N = 3;
f = exp(1);
b = 0.92;
nBins = 30;
width = 1;
binLambdaMin = 0;
binLambdaMax = 14;
% For a constant variance and N=10, use 
% binLambdaMin = 5;
% binLambdaMax = 19;
ensemble = 'ConstantVariance';
[logWeights, nc, f, histogram, accepted, rejected] = metropolisEigenValues(outSteps, inSteps, N, f, b, nBins, width, binLambdaMin, binLambdaMax, ensemble);

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

function bin = computeBin(lambdaC, binLambdaMin, binLambdaMax, nBins)
    bin = round((lambdaC - binLambdaMin) / (binLambdaMax - binLambdaMin) * (nBins-1)) + 1;
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

function [logWeights, nc, f, histogram, accepted, rejected] = metropolisEigenValues(outSteps, inSteps, N, f, b, nBins, width, binLambdaMin, binLambdaMax, ensemble)
    logWeights = zeros(nBins, 1);
    histogram = zeros(nBins, 1);
    nc = 0;
    
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

    accepted = 0;
    rejected = 0;
    mcSteps = 0;
    for outerIdx=1:outSteps
        for innerIdx=1:inSteps
            mcSteps = mcSteps + 1;

            % Check every 10,000 steps to see if we should recalculate the 
            % eigenvectors and values with zeig to be more accurate
            if mod(mcSteps, 10000) == 0
                s = rng;
                [lambda, eigVectors] = zeig(double(full(A)));
                rng(s);
                
                eigVectors = eigVectors(:, lambda > 0);
                lambda = lambda(lambda > 0);   
                lambdaMax = max(lambda);
            end

            indices = randi([1, N], 1, 3);
            proposedJump = randn() * width;
            
            Anew = A;
            Anew(indices) = Anew(indices) + proposedJump;
            
            [lambdaNew, eigVectorsNew] = getEigenvalues(eigVectors, Anew);
            lambdaMaxNew = max(lambdaNew);
            
            if lambdaMaxNew > binLambdaMin && lambdaMaxNew < binLambdaMax
                binLambdaNew = computeBin(lambdaMaxNew, binLambdaMin, binLambdaMax, nBins);
                binLambda = computeBin(lambdaMax, binLambdaMin, binLambdaMax, nBins);
                monteCarloProbability = pijk(Anew(indices), indices) / pijk(A(indices), indices) * exp(logWeights(binLambdaNew) - logWeights(binLambda));

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
        end
        
        disp([accepted, rejected]);

        choppedHistogram = histogram(2:end-1);
        if min(choppedHistogram) > mean(choppedHistogram) * b
            f = sqrt(f);
            nc = nc + 1; 
            writematrix(logWeights, sprintf("./MatlabWeights/Weights_%d_%d_%s.txt", nc, N, ensemble));
            writematrix(histogram, sprintf("./MatlabWeights/Histogram_%d_%d_%s.txt", nc, N, ensemble));
            histogram = histogram * 0;
            disp(nc);
        end
    end
end