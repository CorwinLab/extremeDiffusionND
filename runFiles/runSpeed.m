runTime = [];

for N=2:15
    tic;
    arr = symtensor(@randn, 3, N);
    arr = double(full(arr));
    s = rng;
    [lambda, V, res, cnd] = zeig(arr, 'symmetric');
    rng(s);
    runTime = [runTime toc];
    display(runTime);
end

writematrix(runTime, 'runSpeed.txt');