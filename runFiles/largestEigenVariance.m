function [] = largestEigenVariance(dir, sysID, seed, N, numSamples)
directory = sprintf("%s", dir);
cd(directory);
N = str2num(N);
numSamples = str2num(numSamples);
fileName = sprintf("%s/EigenValues%s.txt", dir, sysID);
eigenValues = zeros(numSamples, 2);

rng(str2num(seed));
display(rng);
for i=1:numSamples
    arr = symtensor(@(x,y) normrnd(0, sqrt(10), x, y), 3, N);
    arr = double(full(arr));
    for i=1:N
        arr(i, i, i) = normrnd(0, 1);
    end
    s = rng;
    [lambda, V, res, cnd] = zeig(arr);
    rng(s);
    eigenValues(i, 1) = lambda(end);
    display(eigenValues(i, 1));
    eigenValues(i, 2) = res(end);
end
writematrix(eigenValues, fileName);
end