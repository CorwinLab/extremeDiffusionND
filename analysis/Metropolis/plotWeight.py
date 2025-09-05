import numpy as np
from matplotlib import pyplot as plt

def weightsToDist(weights, xvals):
    weights = -weights
    weights = weights - np.max(weights)
    binWidth = np.diff(xvals)[0]
    
    dist = np.exp(weights)
    norm = np.sum(dist * binWidth)
    dist /= norm

    mean = np.sum(dist * xvals * binWidth)
    var = np.sum(dist * xvals**2 * binWidth) - mean**2
    std = np.sqrt(var)

    xvals = (xvals - mean) / std 

    return xvals, dist * std

weights = np.loadtxt("./N3LargerWidth/Weights_11_3.txt", dtype=str)
weights = np.array([eval(i) for i in weights])
bins = np.linspace(0, 14, 30)

xvals, dist = weightsToDist(weights, bins)

N10weights = np.loadtxt("Weights_10_10.txt")
N10weights = N10weights
N10bins = np.linspace(4.5, 16, 30)

N10xvals, N10dist = weightsToDist(N10weights, N10bins)

N3EigenVals = np.loadtxt('../ZEigVals/data/allN3EigenValues.txt', delimiter=',')
N3EigenVals = N3EigenVals[:, 0]

N10EigenVals = np.loadtxt('allZEigenvaluesN10.txt', delimiter=',')
N10EigenVals = N10EigenVals[:, 0]

# mean = np.mean(N3EigenVals)
# std = np.std(N3EigenVals)
# normalizedEVals = (N3EigenVals - mean) / std

kpzFit = np.loadtxt("../ZEigVals/data/KPZFit.txt")

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(bins, dist)
ax.plot(N10bins, N10dist, c='b')
ax.hist(N3EigenVals, bins='fd', density=True, histtype='step', color='r')
ax.hist(N10EigenVals, bins='fd', density=True, histtype='step')
# ax.plot(kpzFit[:, 0], kpzFit[:, 1], ls='--', c='k')
fig.savefig("./Figures/Weights.png", bbox_inches='tight')