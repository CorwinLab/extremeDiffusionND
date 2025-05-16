function prob = pijkCyclicVariance(xijk, indices)
    prob = exp(-xijk^2 / 2 * d(indices));
end