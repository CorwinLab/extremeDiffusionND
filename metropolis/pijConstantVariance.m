function prob = pijConstantVariance(xijk, indices)
    prob = exp(-xijk^2 / 2);
end