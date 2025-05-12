from matplotlib import pyplot as plt
import numpy as np
from TracyWidom import TracyWidom

def weightsToDist(weights, xvals):
    weights = -weights
    weights = weights - np.max(weights)
    binWidth = np.diff(xvals)[0]
    
    dist = np.exp(weights)
    norm = np.sum(dist * binWidth)
    dist /= norm 

    return dist

weights = np.loadtxt("./MatlabWeights/Weights_15_3.txt")
xvals = np.linspace(1, 7, 10)
dist = weightsToDist(weights, xvals)

eigvals = np.loadtxt("MaxEigenvalues.txt")

fig, ax = plt.subplots()
ax.plot(xvals, dist)
ax.hist(eigvals, density=True, bins='fd')
fig.savefig("./MatlabWeights/Weights.png", bbox_inches='tight')