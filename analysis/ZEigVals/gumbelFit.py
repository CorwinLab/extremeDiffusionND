from matplotlib import pyplot as plt
import numpy as np

eigVals = np.loadtxt("./data/compiledEigenValues.txt", delimiter=',')

eigVals = eigVals[:, 0]
normalizedEigVals = (eigVals - np.mean(eigVals)) / np.std(eigVals)

data = np.loadtxt("./data/allVar10Eigenvalues.txt", delimiter=',')
data = data[:, 0]
normalizedData = (data - np.mean(data)) / np.std(data)

data = np.loadtxt("./data/allZeroDiagonalEigen.txt", delimiter=',')
data = data[:, 0]
normalizedDataZero = (data - np.mean(data)) / np.std(data)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.hist(normalizedData, bins='fd', density=True, histtype='step', color='b')
ax.hist(normalizedEigVals, bins='fd', density=True, histtype='step', color='g')
ax.hist(normalizedDataZero, bins='fd', density=True, histtype='step', color='r')
fig.savefig("MinEnergy.pdf", bbox_inches='tight')