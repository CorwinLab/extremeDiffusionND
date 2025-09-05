import numpy as np
from matplotlib import pyplot as plt

# eigVals = np.loadtxt("./data/compiledEigenValues.txt", delimiter=',')

# eigVals = eigVals[:, 0]
# normalizedEigVals = (eigVals - np.mean(eigVals)) / np.std(eigVals)

kpzFit = np.loadtxt("./data/KPZFit.txt")
print(kpzFit[:, 0], kpzFit[:, 1])

fig, ax = plt.subplots()
# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.set_xlim([10**-3, 7])
# hvals, bins, _ = ax.hist(normalizedEigVals, density=True, histtype='step', bins='fd')
# plt.clf()
# center = (bins[:-1] + bins[1:]) / 2
# np.savetxt("hvals.txt", np.array([hvals, center]))
data = np.loadtxt("hvals.txt")
hvals = data[0, :]
logHvals = -np.log(hvals)
logHvals -= np.min(logHvals)
# logHvals -= np.min(logHvals)
center = data[1, :]
center -= center[np.argmin(logHvals)]
ax.loglog(center, logHvals, c='b')
ax.loglog(-center, logHvals, c='k')
x = np.geomspace(1e-1, 8)
ax.plot(x,.7*x**2.25)
ax.plot(x,.5*x**1.75)
# ax.plot(x, x**1.125)
fig.savefig("PowerLaws.pdf", bbox_inches='tight')