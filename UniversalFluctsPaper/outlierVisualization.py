from matplotlib import pyplot as plt
import numpy as np
import json
import dataAnalysis as d

def cutoffData(statsList, tMaxList, alpha=1, vlpMin=1e-3):
    """
    takes list of stats files, chops off values where var[lnP] > vlpMax
    and then computes
    std( g - t^alpha ) where g = vlp / ( lambda*v**2 )
    """
    # initialize arrays for concatenate
    scalingFuncs = np.array([])
    vlps = np.array([])
    vs = np.array([])
    times = np.array([])
    ls = np.array([])
    for i in range(len(statsList)):
        print(statsList[i])
        file = statsList[i]
        tempData, label = d.processStats(file)
        for j in range(len(tMaxList)):
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            # chop data above vlpMax
            indices = np.array(np.where((tempData[2, :] == tMaxList[j])
                                        & (tempData[1, :] >= 2)
                                        & (tempData[0,:] > vlpMin))).flatten()
            # pull out the r, t, and lambdas of our masked data, then calc lambda r^2/t^2
            r = tempData[1, indices]  # radii
            t = tempData[2, indices]  # time
            l = tempData[3, indices]  # lambda_ext
            scalingFuncVals = d.masterCurveValue(r, t, l)
            # cast our velocities, assuming r = v t^alpha
            vLin = r / t**alpha

            # smash into list
            scalingFuncs = np.concatenate((scalingFuncs, scalingFuncVals))
            vlps = np.concatenate((vlps, tempData[0,indices]))
            vs = np.concatenate((vs, vLin))
            times = np.concatenate((times, t))
            ls = np.concatenate((ls, l))
    return scalingFuncs, vlps, vs, times, ls


if __name__ == "__main__":
    path003 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.03162278/L5000/tMax10000/StatsNoLog.h5"
    path01 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.1/L5000/tMax10000/StatsNoLog.h5"
    path03 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.31622777/L5000/tMax10000/StatsNoLog.h5"
    path1 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/StatsNoLog.h5"
    path3 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000/StatsNoLog.h5"
    path10 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000/StatsNoLog.h5"
    path31 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000/StatsNoLog.h5"
    pathLogNormal = "/mnt/locustData/memoryEfficientMeasurements/h5data/logNormal/0,1/L5000/tMax10000/StatsNoLog.h5"
    pathDelta = "/mnt/locustData/memoryEfficientMeasurements/h5data/Delta/L5000/tMax10000/StatsNoLog.h5"
    pathCorner = "/mnt/locustData/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000/StatsNoLog.h5"

    # # for all data, all times
    statsFileList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]
    # statsFileList = [path1]
    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)

    tMaxList = np.array(variables['ts'])

    # is it constant?
    alpha = 1
    # should give me x and y, with data chopped off
    vlpMin = 1e-10
    scalingFunc, vlp, vs, times, ls = cutoffData(statsFileList, tMaxList,
                                                     vlpMin=vlpMin, alpha=alpha)

    largeVIndices = (vs > 0.8 )
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
        plt.loglog(scalingFunc[largeVIndices], vlp[largeVIndices], 'o', markeredgecolor='k', ms=4, mew=0.5, zorder=np.random.rand())
        plt.xlabel(r"$\frac{\displaystyle \lambda_{\mathrm{ext}}r^2}{t^2}$")
        plt.ylabel(r"$\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")

        plt.xticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
        plt.yticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])

        savePath = "/home/fransces/Documents/Figures/testfigs/loss/largeVmastercurve.png"
        plt.savefig(f"{savePath}")

        plt.figure(figsize=(5,5), constrained_layout=True, dpi=150)
        plt.semilogx(times[largeVIndices], (1 / (ls[largeVIndices] * vs[largeVIndices] ** 2)) * vlp[largeVIndices], '.', alpha=.05)
        plt.ylim([0,50])
        plt.xlim([1,2e4])
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\frac{\displaystyle 1}{\lambda_{\mathrm{ext}}v^2}\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")
        savepath2 = "/home/fransces/Documents/Figures/testfigs/loss/largeVFlat.png"
        plt.savefig(f"{savepath2}")

        plt.figure(figsize=(5,5), constrained_layout=True, dpi=150)
        plt.loglog(times[largeVIndices], vlp[largeVIndices], '.',ms=2)
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")
        savepath3 = "/home/fransces/Documents/Figures/testfigs/loss/largeVNoScaling.png"
        plt.savefig(f"{savepath3}")
