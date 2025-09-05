import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import json
from TracyWidom import TracyWidom

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

alpha = 1
directory = f'/mnt/talapasShared/MillionQuad2DProbs/{alpha},{alpha},{alpha},{alpha}/'
regime = 'linear'

finalProbsFile = os.path.join(directory, 'FinalProbs.h5')
meanVarFile = os.path.join(directory, 'MeanVar.h5')
vars = os.path.join(directory, 'variables.json')

with h5py.File(meanVarFile, 'r') as f:
    numFiles = f.attrs['numberOfFiles']
    skew = f[regime]['skew'][-1, :]

with open(vars, 'r') as f:
    vars = json.load(f)
    time = vars['ts']
    velocity = vars['velocities']

with h5py.File(finalProbsFile, 'r') as f:
    probs = f[f'temp{regime}']

    specificProbs = probs[:numFiles, -1]

rescaledProbs = (specificProbs - np.mean(specificProbs)) / np.sqrt(np.var(specificProbs))

def gaussian(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * np.sqrt(var)) * np.exp(- 1 / 2 * (x-mean)**2 / var)

xvals = np.linspace(-6, 6, num=1000)
mean = 0
var = 1
kpzFit = np.loadtxt("./Data/KPZFit.txt")

tw1 = TracyWidom(beta=2)
mean = -1.77
var = 0.813
pdf = tw1.pdf(xvals)

fig, ax = plt.subplots()
ax.set_ylim([10**-5, 1])
ax.set_yscale("log")
ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$\mathrm{Probability\ Density}$")
ax.hist(rescaledProbs, bins='fd', histtype='step', density=True)
ax.plot(kpzFit[:, 0], kpzFit[:, 1], ls='--', c='k')
# ax.plot((xvals - mean) / np.sqrt(var), pdf, ls='--', c='r')
fig.savefig("./Figures/LinearRegimeDist.svg", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel(r"$v$")
ax.set_xlim([0, 1])
ax.set_ylabel(r"$\mathrm{Skew}(\xi)$")
ax.scatter(velocity, skew)
fig.savefig("./Figures/LinearSkew.svg", bbox_inches='tight')