import numpy as np
import h5py
import json
from matplotlib import pyplot as plt

def plotBoundaryEffects():
    # only run this in the folder its currently in, ie. /home/fransces/Documents/code/extremeDiffusionND/UniversalFluctsPaper/showBoundaryEffeects/

    i = -5  # idx to pull out v = 0.63 or something

    with open("variables.json","r") as v:
        variables = json.load(v)
    times = variables['ts']
    velocities = variables['velocities']

    # L = 200
    a0 = h5py.File('20.h5', 'r')['regimes']['linear'][:, i]
    # L = 500, tMax = 1000
    a = h5py.File("18.h5", "r")['regimes']['linear'][:, i]
    # L = 800, tMax = 1000
    b = h5py.File('17.h5',"r")['regimes']['linear'][:, i]
    # L = 1000, tMax = 1000
    d = h5py.File('15.h5','r')['regimes']['linear'][:, i]

    allData = [a0, a, b, d]
    labels = [r"$L = 200 < t_{max} $",
              r"$ L = 500 < t_{max}$",
              r"$ L = 800 < t_{max}$",
              r"no boundary"]
    plt.ion()
    plt.figure()
    for x in range(len(allData)):
        plt.loglog(times, allData[x], label=labels[x], alpha=0.8)
    plt.xlabel(r"$t$")
    plt.ylabel("probabilities")
    plt.title(f"Dirichlet, alpha = 1, v = {velocities[i]}, tMax = 1000")
    plt.legend()
    plt.savefig("rawProbComparison.png")

    plt.figure()
    # calc. res wrt "d"
    ylabel = r"$2 |\mathbb{P}_{L} - \mathbb{P}_{\mathrm{no boundary}}| / \left(\mathbb{P}_{L} + \mathbb{P}_{\mathrm{no boundary}}\right)$"
    for x in range(len(allData[:3])):
        res = 2 * (np.abs(allData[x] - d)) / (allData[x] + d)
        plt.plot(times, res, label=labels[x],alpha=0.8)
    plt.xlabel(r"$t$")
    plt.ylabel(ylabel)
    plt.title(f"Dirichlet, alpha = 1, v = {velocities[i]}, tMax = 1000")
    plt.legend()
    plt.savefig("residueComparison.png")

    plt.figure()
    for x in range(len(allData[:3])):
        res = 2 * (np.abs(allData[x] - d)) / (allData[x] + d)
        plt.semilogy(times, res, label=labels[x],alpha=0.8)
    plt.xlabel(r"$t$")
    plt.ylabel(ylabel)
    plt.title(f"Dirichlet, alpha = 1, v = {velocities[i]}, tMax=1000")
    plt.legend()
    plt.savefig('resudieComparisonSemiLog.png')