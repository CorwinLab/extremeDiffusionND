function count = d(indices)
    i = indices(1);
    j = indices(2);
    k = indices(3);
   
    if i == j && j==k
        count = 1;
    elseif i == j || j == k || i == k
        count = 3;
    else 
        count = 6;
    end
end