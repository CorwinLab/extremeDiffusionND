function [] = largestEigenValue(dir, sysID, seed, N, numSamples)
directory = sprintf("%s", dir);
cd(directory);
N = str2num(N);
numSamples = str2num(numSamples);
fileName = sprintf("%s/EigenValues%s.txt", dir, sysID);
eigenValues = zeros(numSamples, 2);

rng(str2num(seed));

for i=1:numSamples
    arr = symtensor(@randn, 3, N);
    arr = double(full(arr));
    s = rng;
    [lambda, V, res, cnd] = zeig(arr, 'symmetric');
    display(i);
    rng(s);
    eigenValues(i, 1) = lambda(end);
    eigenValues(i, 2) = res(end);
end

writematrix(eigenValues, fileName);
end
