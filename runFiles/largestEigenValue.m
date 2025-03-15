function [] = largestEigenValue(dir, sysID)
directory = sprintf("%s", dir)
cd(directory);
N = 10;
numSamples = 50;
fileName = sprintf("%s/EigenValues%s.txt", dir, sysID);
eigenValues = zeros(numSamples, 1);

rng(str2num(sysID));

for i=1:numSamples
    arr = symtensor(@randn, 3, N);
    arr = double(full(arr));
    s = rng;
    lambda = eeig(arr);
    rng(s);
    realLambda = lambda(abs(imag(lambda)) < 10 * eps);
    maxLambda = max(real(realLambda));
    eigenValues(i) = maxLambda;
end

writematrix(eigenValues, fileName);
end
