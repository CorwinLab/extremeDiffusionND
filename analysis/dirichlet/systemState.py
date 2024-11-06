import numpy as np 
from matplotlib import pyplot as plt
import copy
import matplotlib
import sys 
sys.path.append("../../")
from memEfficientEvolve2DLattice import calculateRadii, tOnSqrtLogT

data = './Data/6111.npz'
data = np.load(data)

probs = np.load('./Data/459.npy')
info = np.load("./Data/info.npz")
times = info['times']

probs = probs[0]
probs_smallest_velocity = probs[:, 0]
fig, ax = plt.subplots()
ax.plot(times, probs_smallest_velocity)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("t")
ax.set_ylabel("log(prob)")
fig.savefig("./Figures/ProbsSmallest.png")


t = 6124
vs = np.geomspace(10 ** (-3), 10, 21)
Rs = vs * t / np.sqrt(np.log(t))

occ = data['arr_0']

cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
vmax = np.max(occ)
vmin = vmax / 1e3

fig, ax = plt.subplots()
ax.set_xlim([4500, 5500])
ax.set_ylim([4500, 5500])
ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap, interpolation='none')
fig.savefig("./Figures/Occupancy.pdf", bbox_inches='tight')