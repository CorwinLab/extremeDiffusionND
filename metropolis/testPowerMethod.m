N = 10;

symTensor = symtensor(@randn, 3, N);
tensor = double(full(symTensor));
[lambda, V] = zeig(tensor);
maxLambda = lambda(end);
maxEigenVector = V(:, end);

eigValues = zeros(length(lambda), 1);
eigVectors = zeros(length(lambda), N);
for eigIdx=1:length(lambda)
    [powerLambda, powerV, flag, iters] = eig_sshopm(full(symTensor), 'Start', V(:, eigIdx));
    eigValues(eigIdx) = powerLambda;
    eigVectors(eigIdx, :) = powerV;
end
% Take maximum of all power method
maxPowerLambda = max(eigValues);

numSamples = 0;

while abs(maxPowerLambda - maxLambda) < 1e-10
    jump = randn();
    index = randi([1 N], 1, 3);
    symTensor(index) = symTensor(index) + jump;
    
    % Get eigenvalues for all starting eigenvectors
    tic
    for eigIdx=1:length(eigValues)
        [powerLambda, powerV, flag, iters] = eig_sshopm(full(symTensor), 'Start', transpose(eigVectors(eigIdx, :)));
        eigValues(eigIdx) = powerLambda;
        eigVectors(eigIdx, :) = powerV;
    end
    % Take maximum of all power method
    maxPowerLambda = max(eigValues);
    toc
    
    tic
    [lambda, V] = zeig(double(full(symTensor)));
    maxLambda = lambda(end);
    toc

    numSamples = numSamples + 1;
    disp(numSamples);
end

display(numSamples);