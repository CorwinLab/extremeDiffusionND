function [] = largestEigenValueHermitian(dir, sysID)
directory = sprintf("%s", dir);
cd(directory);

N = 3;
numSamples = 3;
fileName = sprintf("%s/EigenValues%s.txt", dir, sysID);
eigenValues = zeros(numSamples, 1);
rng(str2num(sysID));

A = normrnd(0, 1/2, N, N, N, numSamples) + 1i * normrnd(0, 1/2, N, N, N, numSamples);

C = permute(A, [1 2 3 4]) + conj(permute(A, [2 1 3 4])) + (permute(A, [2 3 1 4])) + conj(permute(A, [3 2 1 4])) + (permute(A, [3 1 2 4])) + conj(permute(A, [1 3 2 4]));
C = C / 6;

for i=1:N
    C(i, i, i, :) = normrnd(0, 1, 1, numSamples);
end

for i=1:numSamples
    arr = C(:, :, :, i);
    lambda = eeig(arr);
    disp(lambda);
    realLambda = lambda(abs(imag(lambda)) < 10 * eps);
    maxLambda = max(real(realLambda));
    eigenValues(i) = maxLambda;
end

writematrix(eigenValues, fileName);
end