import numpy as np
from matplotlib import pyplot as plt

mean = np.loadtxt("./MeanVar/Mean.txt")
var = np.loadtxt("./MeanVar/Variance.txt")
Ns = np.loadtxt("./MeanVar/NValues.txt")

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(Ns, mean)
ax.plot(Ns, np.sqrt(2 * Ns), ls='--')
fig.savefig("./Figures/Mean.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(Ns, var)
fig.savefig("./Figures/Var.png", bbox_inches='tight')