function A = getCyclicTensor(N)
    A = symtensor(@randn, 3, N);

    for i=1:N
        for j=1:i
            for k=1:j
                if i==j && j==k
                    continue
                elseif i==j || i==k || j==k
                    A(i, j, k) = A(i, j, k) / sqrt(3);
                else
                    A(i, j, k) = A(i, j, k) / sqrt(6);
                end
            end
        end
    end 
end