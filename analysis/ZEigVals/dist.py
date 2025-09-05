import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skew
from TracyWidom import TracyWidom

eigVals = np.loadtxt("./data/compiledEigenValues.txt", delimiter=',')

eigVals = eigVals[:, 0]
normalizedEigVals = (eigVals - np.mean(eigVals)) / np.std(eigVals)

kpzFit = np.loadtxt("./data/KPZFit.txt")

N3EigenVals = np.loadtxt('./data/allN3EigenValues.txt', delimiter=',')
N3EigenVals = N3EigenVals[:, 0]
N3EigenVals = (N3EigenVals - np.mean(N3EigenVals)) / np.std(N3EigenVals)

N15EigenVals = np.loadtxt('./data/mathAll15.txt')
N15EigenVals = (N15EigenVals - np.mean(N15EigenVals)) / np.std(N15EigenVals)

xvals = np.linspace(-5, 10)

tw = TracyWidom(beta=1)
meanTW = -1.77
stdTW = np.sqrt(0.813)

def gumbel(xvals, mu, beta):
    z = (xvals - mu) / beta
    return 1 / beta * np.exp(-(z + np.exp(-z))) 

scale = np.sqrt(6 / np.pi**2)
loc = -scale * 0.577

xvals = np.linspace(2, 6)

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_ylim([10**-5, 1])
ax.hist(normalizedEigVals, bins='fd', density=True, histtype='step', color='b')
ax.hist(N3EigenVals, bins='fd', density=True, histtype='step', color='r')
ax.plot(kpzFit[:, 0], kpzFit[:, 1], ls='--', c='k')
# ax.plot(xvals, gumbel(xvals, loc, scale), ls='--', color='m')
# ax.hist(N15EigenVals, bins='fd', density=True, histtype='step', color='g')
ax.plot(xvals, np.exp(-xvals), ls='--')


fig.savefig("Dist.pdf", bbox_inches='tight')