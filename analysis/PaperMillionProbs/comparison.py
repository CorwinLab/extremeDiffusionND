import npquad
import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import json

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

directory = f'/mnt/corwinLab/2DData/PaperMillionQuad2DProbs/'

finalProbsFile = os.path.join('/home/jacob/Desktop/', 'FinalProbs.h5')
meanVarFile = os.path.join('/home/jacob/Desktop/', 'MeanVar.h5')

vars = os.path.join('/home/jacob/Desktop/', 'variables.json')

with h5py.File(meanVarFile, 'r') as f:
    numFiles = f.attrs['numberOfFiles']
    print(numFiles)

with open(vars, 'r') as f:
    vars = json.load(f)
    time = vars['ts']
    velocity = vars['velocities']
    print(velocity[-1])

with h5py.File(finalProbsFile, 'r') as f:
    probs = f[f'tempsqrt']
    specificProbs = probs[:numFiles, -1]

    probsLarge = f['templinear']
    largeProbs = probsLarge[:numFiles, -1]

rescaledProbs = (specificProbs - np.mean(specificProbs)) / np.std(specificProbs)
rescaledLinearProbs = (largeProbs - np.mean(largeProbs)) / np.std(largeProbs)

def gaussian(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * np.sqrt(var)) * np.exp(- 1 / 2 * (x-mean)**2 / var)

xvals = np.linspace(-5, 5, num=1000)
mean = 0
var = 1

kpzFit = np.loadtxt("./conerDist/KPZFit.txt")
cornerDist = np.loadtxt("./conerDist/FinalProbs.txt")
cornerDist = (cornerDist - np.mean(cornerDist)) / np.std(cornerDist)

bins = 200
fig, ax = plt.subplots()
ax.set_ylim([10**-5, 1])
ax.set_xlim([-5, 7])
ax.set_yscale("log")
ax.set_xlabel(r"$\displaystyle\frac{\log\left(X(t)\right) - \mu^{\bm{\xi}}(t)}{\sigma^{\bm{\xi}}(t)}$")
ax.set_ylabel(r"$\mathrm{Probability\ Density}$")

# Dirichlet binning
# rescaledProbs = np.sort(rescaledProbs)
# rescaledProbsBins = rescaledProbs[::1000]

def dirichletBinning(data, dirichletN=500, minWidth=0.05):
    data = np.sort(data)
    bins = data[::dirichletN]
    # Group neighboring bins together if their width is smaller than some ammount
    lowEdge = bins[0]
    newBins = [lowEdge]
    for edge in bins[1:-1]:
        if (edge - lowEdge) > minWidth:
            newBins.append(edge)
            lowEdge = edge
    newBins.append(bins[-1])
    return newBins

ax.hist(rescaledProbs, bins=dirichletBinning(rescaledProbs), histtype='step', density=True, color='tab:red', alpha=0.75, label=r'$r(t) \propto \sqrt{t}$')
ax.hist(rescaledLinearProbs, bins=dirichletBinning(rescaledLinearProbs), histtype='step', density=True, color='tab:blue', alpha=0.75, label=r'$r(t) \propto t$')
ax.hist(cornerDist, bins=dirichletBinning(cornerDist), histtype='step', density=True, color='tab:green', alpha=0.75, label=r'$r(t) = t$')
ax.plot(xvals, gaussian(xvals, mean, var), ls='-.', c='r')
ax.plot(kpzFit[:, 0], kpzFit[:, 1], ls='--', c='blue')

leg = ax.legend(
    loc="upper right",
    framealpha=0,
    labelcolor=["r", "b", 'g'],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)

fig.savefig("./DistComp.svg", bbox_inches='tight')
