import dataAnalysis as d
from matplotlib import pyplot as plt
import numpy as np
import json

if __name__ == "__main__":
    path003 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.03162278/L5000/tMax10000/Stats.h5"
    path01 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.1/L5000/tMax10000/Stats.h5"
    path03 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.31622777/L5000/tMax10000/Stats.h5"
    path1 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/Stats.h5"
    path3 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000/Stats.h5"
    path10 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000/Stats.h5"
    path31 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000/Stats.h5"
    pathLogNormal = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/logNormal/0,1/L5000/tMax10000/Stats.h5"
    pathDelta = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/Delta/L5000/tMax10000/Stats.h5"
    pathCorner = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000/Stats.h5"

    # for all data, all times
    statsFileList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]
    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)

    tMaxList = np.array(variables['ts'])

    # is it constant?
    alpha = 1
    # should give me x and y, with data chopped off
    scalingFunc, vlp, vs, times, ls = d.prepLossFunc(statsFileList, tMaxList,
                                                     vlpMax=1e-3,alpha=alpha)
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    plot = True
    if plot:
        # plotting to make sure data is chopped
        plt.ion()
        plt.figure(figsize=(5, 5), constrained_layout=True, dpi=150)
        plt.xlim([1e-11, 5e2])
        plt.ylim([1e-11, 5e2])
        plt.gca().set_aspect('equal')
        plt.loglog(scalingFunc, vlp, 'o', markeredgecolor='k', ms=4, mew=0.5, zorder=np.random.rand())
        plt.xlabel(r"$\frac{\displaystyle \lambda_{\mathrm{ext}}r^2}{t^2}$")
        plt.ylabel(r"$\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")

        plt.xticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
        plt.yticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])

        savePath = "/home/fransces/Documents/Figures/testfigs/loss/moreChoppedData.png"
        plt.savefig(f"{savePath}")

        plt.figure(figsize=(5,5), constrained_layout=True, dpi=150)
        plt.semilogx(times, (1 / (ls * vs ** 2)) * vlp, '.', alpha=.05)
        plt.ylim([0,4/3])
        plt.xlim([1,2e4])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\frac{\displaystyle 1}{\lambda_{\mathrm{ext}}v^2}\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")
        savepath2 = "/home/fransces/Documents/Figures/testfigs/loss/moreChoppedDataFlat.png"
        plt.savefig(f"{savepath2}")

    # actual loss func
    g = vlp / (ls * vs**2)
    print(f"mean g, std g: {np.mean(g), np.std(g)}")
    alphas = np.linspace(-2,2,1001)
    s = [np.std(g - times**alpha) for alpha in alphas]
    plt.figure(figsize=(5,5),constrained_layout=True,dpi=150)
    # plt.plot(alphas,var)
    plt.semilogy(alphas, s)
    minAlpha = alphas[np.argmin(s)]
    plt.vlines(minAlpha, 0, 1,color='black')
    plt.title(f"black = {minAlpha}")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\sqrt{\langle (g - At^\alpha - \langle g\rangle - \langle A t^\alpha \rangle )^2 \rangle}$")
    savepath3 = "/home/fransces/Documents/Figures/testfigs/loss/moreChoppedLoss.png"
    plt.savefig(f"{savepath3}")
    # plot a binned mean (or median?)
    tedge = np.geomspace(5, 1e4)
    binnedMeanG = [np.mean(g[(times > tedge[i]) * (times < tedge[i+1])]) for i in range(len(tedge)-1)]
    binnedMedianG = [np.median(g[(times > tedge[i]) * (times < tedge[i+1])]) for i in range(len(tedge)-1)]
    plt.figure(figsize=(5,5), constrained_layout=True, dpi=150)
    plt.semilogx(tedge[1:], binnedMeanG, label="binned mean g", color='green')
    plt.semilogx(tedge[1:], binnedMedianG, label="binned median g",color='blue')
    plt.xlabel(r"$t$")
    plt.legend()
    plt.savefig("/home/fransces/Documents/Figures/testfigs/loss/moreChoppedBinned.png")


