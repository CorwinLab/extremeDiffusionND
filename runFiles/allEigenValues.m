function [] = allEigenValues(dir, sysID, seed, N, numSamples)
directory = sprintf("%s", dir);
cd(directory);
N = str2num(N);
numSamples = str2num(numSamples);
fileName = sprintf("%s/EigenValues%s.txt", dir, sysID);
eigenValues = [];

rng(str2num(seed));

for i=1:numSamples
    tic;
    arr = symtensor(@randn, 3, N);
    arr = double(full(arr));
    s = rng;
    lambda = zeig(arr);
    eigenValues = [eigenValues lambda];
    rng(s);
    toc;
end
writematrix(transpose(eigenValues), fileName);
end
