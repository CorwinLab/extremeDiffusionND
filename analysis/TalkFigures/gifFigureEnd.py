import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../../")
from memEfficientEvolve2DLattice import updateOccupancy
import matplotlib 
import copy
from randNumberGeneration import getRandomDistribution
from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({'font.size': 20, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})

maxT = 500
func = getRandomDistribution('LogNormal', [1, 1, 1, 1])

L = 500
occ = np.zeros((2 * L + 1, 2 * L + 1))
occ[L, L] = 1

cmap = copy.copy(matplotlib.cm.get_cmap("jet"))
cmap.set_under(color="white")
cmap.set_bad(color="white")
limits = [500 - 100, 500 + 100]

for t in tqdm(range(maxT)):
    occ = updateOccupancy(occ, t, func)

vmax = 10**-3
vmin = 1e-10

fig, ax = plt.subplots()
im = ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, interpolation='none', alpha=0.75)

ax.set_xlim(limits)
ax.set_ylim(limits)

r = 0.4 * t**(5/6)

circle = plt.Circle((L, L), r, color='k', fill=False, lw=2.5, alpha=0.75)
ax.add_patch(circle)

mutation_scale = 40 * ( 1 - np.exp(-t / 100))

arrow = FancyArrowPatch((L, L), (L + r * np.cos(np.pi / 4), L + r * np.sin(np.pi / 4)), color='k', mutation_scale=mutation_scale)
ax.add_patch(arrow)

ax.annotate(r"$R(t)$", (500, 545))

xvals = np.linspace(400, 600)
yvals = np.sqrt(r**2 - (xvals-L)**2) + L
yvals[np.isnan(yvals)] = 500
y2vals = np.ones(len(xvals)) * 600

ax.fill_between(xvals, yvals, y2vals, alpha=0.75, hatch='//', facecolor='none', edgecolor='k', linewidth=0)
ax.fill_between(xvals, 2 * L - yvals, np.ones(len(xvals)) * 400, alpha=0.75, hatch='//', facecolor='none', edgecolor='k', linewidth=0)

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad = 0.01, ticks=[10**-3, 10**-5, 10**-7, 10**-9])
cbar.ax.set_ylabel(r"$\mathrm{Probability}$", rotation = 270, labelpad=20)
cbar.ax.set_yticklabels([r'$10^{-3}$', r'$10^{-5}$', r'$10^{-7}$', r'$10^{-9}$'], fontdict={'fontsize': 20})

fig.savefig(f"./Figures/PosterViz.svg", bbox_inches='tight')

plt.close(fig)