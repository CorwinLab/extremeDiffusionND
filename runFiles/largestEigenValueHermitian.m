function [] = largestEigenValueHermitian(dir, sysID, seed, N, numSamples)
directory = sprintf("%s", dir);
cd(directory);
N = str2num(N);
numSamples = str2num(numSamples);
fileName = sprintf("%s/EigenValues%s.txt", dir, sysID);
eigenValues = zeros(numSamples, 2);

rng(str2num(seed));

A = normrnd(0, 1/2, N, N, numSamples) + 1i * normrnd(0, 1/2, N, N, numSamples);

C = permute(A, [1 2 3]) + conj(permute(A, [2 1 3]));

for i=1:N
    C(i, i, :) = normrnd(0, 1, 1, numSamples);
end

for i=1:numSamples
    arr = C(:, :, i);
    display(arr);
    [lambda, V, res, cnd] = zeig(arr);
    eigenValues(i, 1) = lambda(end);
    eigenValues(i, 2) = res(end);
end

writematrix(eigenValues, fileName);
end
