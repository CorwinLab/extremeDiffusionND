N = 3;
numSamples = 1000000;
fileName = "RandomEigenValues.txt";
eigenValues = zeros(numSamples, 3);

C = normrnd(0, 1, N, N, N, numSamples);

for i=1:numSamples
    arr = C(:, :, :, i);
    [lambda, V, res, cnd] = zeig(arr);
    try
        eigenValues(i, 1) = lambda(end);
        eigenValues(i, 2) = res(end);
        eigenValues(i, 3) = length(lambda);
    catch
        display("No eigenvalues found:");
        display(i);
        continue
    end
end

writematrix(eigenValues, fileName);
