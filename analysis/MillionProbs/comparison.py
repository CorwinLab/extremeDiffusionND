import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import json

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

alpha = 1
directory = f'/mnt/corwinLab/2DData/MillionQuad2DProbs/{alpha},{alpha},{alpha},{alpha}/'
regime = 'sqrt'

finalProbsFile = os.path.join(directory, 'FinalProbs.h5')
meanVarFile = os.path.join(directory, 'MeanVar.h5')
vars = os.path.join(directory, 'variables.json')

with h5py.File(meanVarFile, 'r') as f:
    numFiles = f.attrs['numberOfFiles']
    print(numFiles)

with open(vars, 'r') as f:
    vars = json.load(f)
    time = vars['ts']
    velocity = vars['velocities']
    print(velocity[-1])

with h5py.File(finalProbsFile, 'r') as f:
    probs = f[f'temp{regime}']
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

kpzFit = np.loadtxt("./Data/KPZFit.txt")

bins = 200
fig, ax = plt.subplots()
ax.set_ylim([10**-5, 1])
ax.set_xlim([-5, 7])
ax.set_yscale("log")
ax.set_xlabel(r"$\displaystyle\frac{\log\left(X(t)\right) - \mu(t)}{\sigma(t)}$")
ax.set_ylabel(r"$\mathrm{Probability\ Density}$")
ax.hist(rescaledProbs, bins=bins, histtype='step', density=True, color='tab:red', alpha=0.75, label=r'$R(t) \propto \sqrt{t}$')
ax.hist(rescaledLinearProbs, bins=bins, histtype='step', density=True, color='tab:blue', alpha=0.75, label=r'$R(t) \propto t$')
ax.plot(xvals, gaussian(xvals, mean, var), ls='-.', c='r')
ax.plot(kpzFit[:, 0], kpzFit[:, 1], ls='--', c='blue')

leg = ax.legend(
    loc="upper right",
    framealpha=0,
    labelcolor=["r", "b"],
    handlelength=0,
    handletextpad=0,
)
for item in leg.legendHandles:
    item.set_visible(False)


fig.savefig("./Figures/DistComp.svg", bbox_inches='tight')
