import h5py
import numpy as np
import os
import json
from matplotlib import pyplot as plt

if __name__ == "__main__":
    statsFile ="/mnt/talapasData/data/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000/Stats.h5"
    stats = h5py.File(f"{statsFile}", "r")
    # for concatenate
    radii = np.array([])
    times = np.array([])
    # setup for grabbing list of vs and ts
    filePath = os.path.split(statsFile)[0]  #gets rid of the Stats.h5
    colors = ['palevioletred','firebrick','darkorange']
    regimes = ['linear', 'tOnSqrtLogT', 'sqrt']
    with open(f"{filePath}/variables.json", "r") as v:
        variables = json.load(v)
    ts = np.array(variables['ts'])
    vs = np.array(variables['velocities'])
    longTs = np.tile(ts,vs.shape[0])  # turn ts array into 1d array of size (336 * 221)
    rMin = 3

    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    fig, ax = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=150,subplot_kw=dict(box_aspect=1))
    with h5py.File(f"{filePath}/0.h5", "r") as f:
        for regime in regimes:
            # ax.loglog(longTs, f['regimes'][regime].attrs['radii'].flatten(order='F'),'.',ms=2,color=colors[regimes.index(regime)],label=regime)
            # (336*21 = 7056 1d array) of linear radii
            # flatten in column major so its like (v[0] radii for all t, v[1] radii for all t, ... etc)
            tempR = f['regimes'][regime].attrs['radii'].flatten(order='F')
            indices = (tempR >= rMin) & (tempR <= longTs)
            radii = np.concatenate((radii, tempR[indices]))
            times = np.concatenate((times, longTs[indices]))
    x = np.logspace(0, 4)
    # ax.legend()
    ax.scatter(times, radii, s=1, color='k',marker='.')
    ax.plot(x, x, color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$r(t)$")
    savepath = "/home/fransces/Documents/Figures/testfigs/radiiScatterPlot.pdf"
    plt.savefig(f"{savepath}")