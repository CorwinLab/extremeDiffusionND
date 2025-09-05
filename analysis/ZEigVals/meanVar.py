import numpy as np
from matplotlib import pyplot as plt

eigVals = np.loadtxt("./data/compiledEigenValues.txt", delimiter=',')
eigVals = eigVals[:, 0]

N3EigenVals = np.loadtxt('./data/allN3EigenValues.txt', delimiter=',')
N3EigenVals = N3EigenVals[:, 0]

N15EigenVals = np.loadtxt('./data/mathAll15.txt')

means = np.array([np.mean(N3EigenVals), np.mean(eigVals), np.mean(N15EigenVals)])
var = np.array([np.var(N3EigenVals), np.var(eigVals), np.var(N15EigenVals)])
Ns = np.array([3, 10, 15])

fig, ax = plt.subplots()
ax.plot(Ns, means)
fig.savefig("./Figures/Means.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(Ns, var)
fig.savefig("./Figures/Variance.png", bbox_inches='tight')