import numpy as np
from matplotlib import pyplot as plt

def weightsToDist(weights, xvals):
    weights = -weights
    weights = weights - np.max(weights)
    binWidth = np.diff(xvals)[0]
    print(weights)
    dist = np.exp(weights)
    norm = np.sum(dist * binWidth)
    dist /= norm

    mean = np.sum(dist * xvals * binWidth)
    var = np.sum(dist * xvals**2 * binWidth) - mean**2
    std = np.sqrt(var)

    xvals = (xvals - mean) / std 
    
    return xvals, dist * std

weightsFile = '/mnt/corwinLab/Code/extremeDiffusionND/metropolis/ConstantVariance/10/MatlabWeights/Weights_9_10_ConstantVariance.txt'
weights = np.loadtxt(weightsFile)
bins = np.linspace(5, 19, 30)
xvals, dist = weightsToDist(weights, bins)

weights = np.loadtxt("../Metropolis/N3LargerWidth/Weights_8_3.txt", dtype=str)
weights = np.array([eval(i) for i in weights])
bins = np.linspace(0, 14, 30)
xvals, dist = weightsToDist(weights, bins)

kpzFit = np.loadtxt("../ZEigVals/data/KPZFit.txt")

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(xvals, dist, c='b')
ax.plot(kpzFit[:, 0], kpzFit[:, 1], ls='--', c='k')
fig.savefig("Dist.png")