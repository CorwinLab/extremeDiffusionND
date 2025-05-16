function A = getConstantVarianceTensor(N)
    A = symtensor(@randn, 3, N);
end