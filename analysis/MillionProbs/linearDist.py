import numpy as np
from scipy.stats import skew
import npquad 
from matplotlib import pyplot as plt
import json
import os
import h5py
plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

alpha = 0.01
directory = f'/mnt/talapasShared/MillionQuad2DProbs/{alpha},{alpha},{alpha},{alpha}/'
file = os.path.join(directory, 'FinalProbs.h5')
variables = os.path.join(directory, 'variables.json')
meanVarFile = f"/mnt/talapasShared/MillionQuad2DProbs/{alpha},{alpha},{alpha},{alpha}/MeanVar.h5"

with h5py.File(meanVarFile, 'r') as f:
    numFiles = f.attrs['numberOfFiles']

with h5py.File(file, 'r') as f:
    linearvals = f['templinear'][:][:numFiles, :]
    peakSkew = linearvals[:, 6]
    endSkew = linearvals[:, -5] # Changed from -2 to -5 for alpha=0.01

with open(variables, 'r') as f:
    vars = json.load(f)

alphas = np.array(vars['params'])
velocities = np.array(vars['velocities'])
times = np.array(vars['ts'])

peakDist = (peakSkew - np.mean(peakSkew)) / np.sqrt(np.var(peakSkew))
endSkew = (endSkew - np.mean(endSkew)) / np.sqrt(np.var(endSkew))

kpzFit = np.loadtxt("./Data/KPZFit.txt")

fig, ax = plt.subplots()
ax.set_yscale('log')
ax.set_xlabel(r"$\ln\left(\mathbb{P}\left(\Vert S(t) \Vert > vt \right) \right)$")
ax.set_ylabel(r"$\mathrm{Probability\ Density}$")
ax.set_ylim([10**-5, 1])
# ax.hist(peakDist, density=True, alpha=0.5, color='tab:green', bins=100, histtype='step')
ax.hist(endSkew, density=True, alpha=0.5, color='tab:red', bins='fd', histtype='step')
ax.plot(kpzFit[:, 0], kpzFit[:, 1], ls='--', c='k')
fig.savefig(f"Dist{alpha}.svg", bbox_inches='tight')