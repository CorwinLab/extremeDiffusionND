ensemble = 'Cyclic';
N = 3;
width = 1/sqrt(6);

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

allArrays = cell(num_samples, 1);
num_samples = 100000;
avgArray = symtensor(@zeros, 3, N);
avgArraySquared = symtensor(@zeros, 3, N);

for i=1:num_samples
    indices = randi([1, N], 1, 3);
    proposedJump = randn() * width;

    Anew = A;
    Anew(indices) = Anew(indices) + proposedJump;
    monteCarloProbability = pijk(Anew(indices), indices) / pijk(A(indices), indices);

    if rand < monteCarloProbability
        A = Anew;
    end
    allArrays{i} = A;
    avgArray = avgArray + A;
    avgArraySquared = avgArraySquared + A.^2;
end

avgArray = avgArray / num_samples;
avgArraySquared = avgArraySquared / num_samples;

varArray = avgArraySquared - avgArray.^2;
display(avgArray);
display(varArray);

