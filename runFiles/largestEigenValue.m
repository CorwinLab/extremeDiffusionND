function [] = largestEigenValue(dir, sysID)
N = 10;
numSamples = 500;
fileName = sprintf("%s/EigenValues%s.txt", dir, sysID);
eigenValues = zeros(numSamples, 1);
A = normrnd(0, 1, N, N, N, numSamples);

for i=1:numSamples
    arr = A(:, :, :, i);
    lambda = eeig(arr);
    realLambda = lambda(abs(imag(lambda)) < 10 * eps);
    maxLambda = max(real(realLambda));
    eigenValues(i) = maxLambda;
    disp(i);
end

writematrix(eigenValues, fileName);
end