import memEfficientEvolve2DLattice as m
import os
import h5py
import json
import matplotlib.patches
from matplotlib import pyplot as plt
from memEfficientEvolve2DLattice import evolve2DDirichlet
import matplotlib
import dataAnalysis as d
import numpy as np
from memEfficientEvolve2DLattice import updateOccupancy
import matplotlib
import copy
from randNumberGeneration import getRandomDistribution
from matplotlib.patches import FancyArrowPatch
from visualizeMeasurements import colorsForLambda

def binnedMedianAll(savePath, statsList, tList, lambdaList, rMin=3):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    fig, ax = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=300)
    fullColors = colorsForLambda(lambdaList)
    tedge = np.geomspace(10, 1e4)
    for i in range(len(statsList)):
        file = statsList[i]
        tempData, label = d.processStats(file)  # label is distribution name
        print(f"{file}")
        # binned median (each nu)
        vlp = tempData[0,:]  # var[ln[P(r(t))]]
        r = tempData[1,:]  # radii
        t = tempData[2,:]  # time
        l = tempData[3,0]  # lambda_ext
        indices = (r >= rMin)
        vLin = r / t**(1)
        # plot data
        # ax.semilogx(t[indices], (1/(l*vLin[indices]**2))*vlp[indices],'.',color=fullColors[i],
        #          alpha=.05,zorder=np.random.rand(), rasterized=True)
        # binned median for each nu
        # use vlp < 1e-3 and r<=2
        scalingFuncAll, vlpAll, vsAll, timesAll, lsAll = d.prepLossFunc([file], tList,
                                                                        vlpMax=1e-3, alpha=1)
        g = vlpAll / (lsAll * vsAll ** 2)
        print(f"nu, mean g, std g: {label, np.mean(g), np.std(g)}")
        binnedMedianG = [np.median(g[(timesAll > tedge[i]) * (timesAll < tedge[i + 1])]) for i in range(len(tedge) - 1)]
        ax.semilogx(tedge[1:], binnedMedianG, color=fullColors[i])
    ax.set_ylim([0,8/3])  # should be a fctor of 2 from the previous one
    ax.set_xlim([1,2e4])
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\displaystyle\frac{ t^2}{r^2 \lambda_{\mathrm{ext}}}\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right)}\right]$")
    fig.savefig(savePath)

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

    savePath = "/home/fransces/Documents/Figures/testfigs/allmedians.png"

    # for all data, all times
    fullList = [path003, path01, path03, path1, path3, path10, path31, pathLogNormal, pathDelta, pathCorner]

    with open("/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tList = np.array(variables['ts'])

    expVarXListFull, lambdaListFull = d.getListOfLambdas(fullList)
    binnedMedianAll(savePath, fullList, tList, lambdaListFull)
