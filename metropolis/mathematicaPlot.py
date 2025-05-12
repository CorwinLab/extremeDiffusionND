import numpy as np
from matplotlib import pyplot as plt

weights = np.loadtxt("./MathematicaWeights/Weights.txt", dtype=str)
weights = np.array([eval(i) for i in weights])
weights = -weights
weights = weights - np.max(weights)

xvals = np.linspace(0, 5, num=100)
binWidth = np.diff(xvals)[0]

dist = np.exp(weights)
norm = np.sum(dist * binWidth)
dist /= norm 

mean = np.sum(dist * xvals * binWidth)
var = np.sum(dist * xvals**2 * binWidth) - mean**2
std = np.sqrt(var)

eigVals = np.loadtxt("./MathematicaWeights/compiledEigenValues.txt", delimiter=',')

eigVals = eigVals[:, 0]
normalizedEigVals = (eigVals - np.mean(eigVals)) / np.std(eigVals)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(xvals, dist )
ax.hist(normalizedEigVals, bins='fd', density=True, histtype='step', color='b')
fig.savefig("./MathematicaWeights/Weights.png", bbox_inches='tight')