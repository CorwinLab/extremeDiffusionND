import numpy as np
from matplotlib import pyplot as plt
from memEfficientEvolve2DLattice import evolve2DLattice
import matplotlib 
import copy

nParticles = 1e8
maxT = 1000
distribution = 'uniform'
pdf = False 
occtype = int 
occupancy = np.array([[nParticles]], dtype=occtype)

for t, occ in evolve2DLattice(occupancy, maxT, distribution, pdf, occtype):
    pass 

cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
vmax = np.max(occ)
vmin = 0.1

fig, ax = plt.subplots()
ax.set_xlim([100, 400])
ax.set_ylim([100, 400])
ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap, interpolation='none')
fig.savefig("Viz.pdf", bbox_inches='tight')