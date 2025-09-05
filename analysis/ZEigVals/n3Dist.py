import numpy as np
from matplotlib import pyplot as plt 

N3EigenVals = np.loadtxt('./data/compiledEigenValues.txt', delimiter=',')
N3EigenVals = N3EigenVals[:, 0]

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.hist(N3EigenVals, bins='fd', density=True, histtype='step', color='r')
fig.savefig("N10.png", bbox_inches='tight')