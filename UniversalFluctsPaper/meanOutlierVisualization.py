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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


def returnSubsetOfData(fileName, rMin, minVar):
    tempData, label = d.processStats(fileName)  # label is distribution name
    indices = np.array(np.where((tempData[1, :] >= rMin) &
                                (tempData[0, :] >= minVar))
                       ).flatten()
    scalingFuncVals = d.masterCurveValue(tempData[1, :][indices], tempData[2, :][indices],
                                         tempData[3, :][indices])
    # dataSubset = tempData[0, :][indices]
    dataSubset = tempData[:,indices]  # 0 is var, 1 is r, 2 is t 3 is lambda?,
    return scalingFuncVals, dataSubset

def masterCurveAndMean(topDir, statsFileList, lambdaList, markers, minVar, verticalLine=True, rMin=3):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True, dpi=300)
    fig2, ax2 = plt.subplots(figsize=(5,5), constrained_layout=True,dpi=300)
    # histogram of dist. to corner
    fig3, ax3 = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=300)
    # plots of rs vs ts?
    fig4, ax4 = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=300)
    # mastercurve (ax2), want subset of data
    ax.set_xlim([1e-11, 5e2])
    ax.set_ylim([1e-11, 5e2])
    ax.set(adjustable='box', aspect='equal')
    minColors = colorsForLambda(lambdaList)
    distToCornerInts = np.array([])

    for i in range(len(statsFileList)):
        # load each file
        print(statsFileList[i])
        file = statsFileList[i]
        # tempData, label = d.processStats(file)  # label is distribution name
        # take data at r>= 3, and then plot only data which ahs var >= varMin
        try:
            reducedXVals, reducedData = returnSubsetOfData(file, rMin, minVar)
            print("shape of reducedData: ", reducedData.shape)

            # for mastercurve (ax)
            if verticalLine:
                # this is theoretical prediction
                xLoc = lambdaList[i]
                ax.loglog([xLoc, xLoc], [xLoc, 5e2], color=minColors[i], linestyle='solid', zorder=0.0000000001)
                # for main collapse (vlp vs masterfunc linear)
            print("mastercurve")
            print("shape of reducedXVals: ",reducedXVals.shape)
            print("shape of reducedData[0,:] : ",reducedData[0,:].shape)
            ax.loglog(reducedXVals, reducedData[0,:], markers[i], color=minColors[i], markeredgecolor='k',
                       ms=4, mew=0.5, zorder=np.random.rand(), rasterized=True)

            # for mean (ax2)
            gaussianbehavior = reducedData[1, :] ** 2 / reducedData[2, :]  # r^2 / t
            print("shape of gaussian behavior: ", gaussianbehavior.shape)
            print("shape of -reducedData[4,:]: ",reducedData[4,:].shape)
            # plot -<lnP> vs r^2/t
            ax2.loglog(gaussianbehavior, -reducedData[4, :], markers[i],
                      color=minColors[i], markeredgecolor='k', ms=4, mew=0.5,
                      zorder=np.random.rand(), rasterized=True)
            vs = reducedData[1, :] / reducedData[2, :]
            # look at distance to corner:
            distToCorner = reducedData[2,:] - reducedData[1,:]  # t - r
            print("dist to Corner: ", distToCorner.astype(int))
            distToCornerInts = np.concatenate((distToCornerInts, distToCorner))
            # print("rs: ", np.sort(np.unique(reducedData[1, :])))
            # print("ts: ", np.sort(np.unique(reducedData[2, :])))
            # print("vs:")
            # print(np.sort(np.unique(vs)))
            print(np.min(vs), np.max(vs))

            # r vs t plot
            ax4.scatter(reducedData[2,:], reducedData[1,:],color=minColors[i],linewidths=0.5,edgecolors='k')
        except Exception as e:
            print("something went wrong with reduced data!")
            continue
    # for var
    ax.set_xlabel(r"$\displaystyle\frac{\lambda_{\mathrm{ext}}r^2}{t^2}$")
    ax.set_ylabel(
        r"$\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right)}\right]$")
    ax.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2])
    ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2])
    fig.savefig(os.path.join(topDir, "isolateDeviationFromGaussianVar.png"))

    # for mean
    x = np.logspace(-4, 3)
    ax2.plot(x, x, color='red')
    ax2.set(adjustable='box', aspect='equal')
    ax2.set_xlabel(r"$r^2 / t$")
    ax2.set_ylabel(
        r"$-\mathbb{E}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right)}\right]$")
    ax2.set_xlim([1e-4, 1e3])
    ax2.set_ylim([1e-4, 1e3])
    ax2.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    ax2.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    fig2.savefig(os.path.join(topDir,"isolateDeviationFromGaussianMean.png"))

    # dist to corner histogram
    distToCornerInts = distToCornerInts.flatten()
    ax3.hist(distToCornerInts,bins=50)
    fig3.savefig(os.path.join(topDir,"isolatedDeviationDistToCornerHistogram.png"))

    # r vs t
    ax4.set_xlabel(r"$t$")
    ax4.set_ylabel(r"$r$")
    x = np.linspace(0,600)
    ax4.plot(x,x,color='red')
    ax4.plot(x, x**(1/2),color='k')
    fig4.savefig(os.path.join(topDir,"rVsT.png"))

# # TODO: fix this because it's not outputting the correct amount of datapoints and incorrectly showing
# # which points are above gaussian
# # maybe filter by if above the gaussian behavior??
# def mean(topDir, statsFileList, tMaxList, lambdaList, markers, rMin=3, minVar=1e10):
#     print("starting mean plot")
#     plt.rcParams.update(
#         {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
#     colors = colorsForLambda(lambdaList)
#     fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True, dpi=300)
#     for i in range(len(statsFileList)):
#         print(statsFileList[i])
#         file = statsFileList[i]
#         # label: distribution name
#         reducedXVals, reducedData = returnSubsetOfData(file)
#         print("shape of reducedData: ", reducedData.shape)
#         with np.errstate(divide='ignore'):
#             # x-axis, r^2/ t
#             gaussianbehavior = reducedData[1, :] ** 2 / reducedData[2, :]
#             # plot -<lnP> vs r^2/t
#             ax.loglog(gaussianbehavior, -reducedData[4, :], markers[i],
#                       color=colors[i], markeredgecolor='k', ms=4, mew=0.5,
#                       zorder=np.random.rand(), rasterized=True)
#             vs = reducedData[1,:] / reducedData[2,:]
#             print("rs: ", np.sort(np.unique(reducedData[1,:])))
#             print("ts: ", np.sort(np.unique(reducedData[2,:])))
#             print("vs:")
#             print(np.sort(np.unique(vs)))
#             print(np.min(vs), np.max(vs))
#     # prediction
#     x = np.logspace(-4, 3)
#     ax.plot(x, x, color='red')
#     ax.set(adjustable='box', aspect='equal')
#     ax.set_xlabel(r"$r^2 / t$")
#     ax.set_ylabel(
#         r"$-\mathbb{E}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right)}\right]$")
#     ax.set_xlim([1e-4, 1e3])
#     ax.set_ylim([1e-4, 1e3])
#     ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
#     ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
#     fig.savefig(os.path.join(topDir,"isolateDeviationFromGaussianMean.png"))
#

if __name__ == "__main__":
    path003 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.03162278/L5000/tMax10000/Stats.h5"
    path01 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.1/L5000/tMax10000/Stats.h5"
    path03 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.31622777/L5000/tMax10000/Stats.h5"
    path1 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/Stats.h5"
    path3 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000/Stats.h5"
    path10 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000/Stats.h5"
    path31 = "/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000/Stats.h5"
    pathLogNormal = "/mnt/locustData/memoryEfficientMeasurements/h5data/logNormal/0,1/L5000/tMax10000/Stats.h5"
    pathDelta = "/mnt/locustData/memoryEfficientMeasurements/h5data/Delta/L5000/tMax10000/Stats.h5"
    pathCorner = "/mnt/locustData/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000/Stats.h5"

    # for all data, all times
    fullList = [path003, path01, path03, path1, path3, path10, path31, pathLogNormal, pathDelta, pathCorner]

    # fullList = [path003, path01, path03, pathDelta]

    # fullList = [path003, path01]
    expVarXListFull, lambdaListFull = d.getListOfLambdas(fullList)
    fullMarkers = ['o'] * 7 + ['D'] + ['v'] + ['s']

    with open("/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])

    topDir = "/home/fransces/Documents/code/extremeDiffusionND/UniversalFluctsPaper/deviationFromGaussian/"
    masterCurveAndMean(topDir, fullList, lambdaListFull, fullMarkers, 1e1)
    # mean(topDir, fullList, tMaxList, lambdaListFull, fullMarkers)
