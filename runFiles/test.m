function [] = test
N = 9;
for i=1:10
    % rng("shuffle")
    arr = symtensor(@randn, N, N);
    arr = double(full(arr));
    who arr
    % oldState = rng().State;
    s = rng;
    lambda = eeig(arr);
    rng(s);
    % rng().State = oldState;
    % realLambda = lambda(abs(imag(lambda)) < 10 * eps);
    % maxLambda = max(real(realLambda));
end