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

# generates gif of RWRE evolution
def generateGifRWRE(occupancy, maxT, alphas, pathName, startT=1, listOfTimes=None):
    os.makedirs(pathName, exist_ok=True)
    func = m.getRandomDistribution("Dirichlet", alphas)
    # L = occupancy.shape[0]//2
    # if listOfTimes is None:
    #     listOfTimes = np.unique(np.geomspace(startT, maxT, num=10).astype(int))
    for t in range(startT, maxT):
        occupancy = m.updateOccupancy(occupancy, t, func)
        # if t in listOfTimes:
        a = np.copy(occupancy)
        if t % 2 == 0:
            if t < 10:
                tStr = f"00{t}"
            elif 10 <= t < 100:
                tStr = f"0{t}"
            else:
                tStr = f"{t}"
            plt.imshow(a)
            plt.savefig(pathName + tStr + '.png', bbox_inches='tight')

# generates gif of SSRW evolution
def generateGifSSRW(occupancy, maxT, pathName):
    """
    example:
        L = 100; tMaX = 500; alphas = np.array([0.5]*4);occ=np.zeros((2*L+1,2*L+1));occ[L,L] = 1
        v.generateGifSSRW(occ, tMaX)
    """
    os.makedirs(pathName, exist_ok=True)
    func = m.getRandomDistribution("OneQuarter", "")
    # L = occupancy.shape[0]//2
    for t in range(1, maxT):
        occupancy = m.updateOccupancy(occupancy, t, func)
        a = np.copy(occupancy)
        if t % 2 == 0:
            if t < 10:
                tStr = f"00{t}"
            elif 10 <= t < 100:
                tStr = f"0{t}"
            else:
                tStr = f"{t}"
            plt.imshow(a)
            plt.savefig(pathName + tStr + ".png", bbox_inches='tight')


# evolved PDF with circle labeled r(t), hatched outside, with colorbar
def measurementPastCircleGraphic():
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    maxT = 500
    func = getRandomDistribution('Dirichlet', [1, 1, 1, 1])

    L = 500
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1

    for t in range(maxT):
        occ = updateOccupancy(occ, t, func)

    # after evolution, normalize by its max so the brightest
    # occ /= np.max(occ)

    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under(color="white")
    cmap.set_bad(color="white")
    # to fix colorization of model graphic ? make them all use the same min and max.
    # vmax = 1
    vmax = np.max(occ)
    vmin = 1e-10

    fig, ax = plt.subplots(figsize=(5,5))
    a = ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, interpolation='none',alpha=1)
    limits = [L - 100, L + 100]
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    r = 75
    circle = plt.Circle((L, L), r, color='k', ls='--', fill=False, lw=2.5)
    ax.add_patch(circle)
    arrow = FancyArrowPatch((L, L), (L + r * np.cos(np.pi / 4), L + r * np.sin(np.pi / 4)), color='k',
                            mutation_scale=40)
    ax.add_patch(arrow)

    ax.annotate(r"$r(t)$", (510, 545))
    # ax.annotate("(b)",xy=(0,1), xycoords='axes fraction', xytext=(+0.5,-0.5),
    #              textcoords='offset fontsize', verticalalignment='top',
    #              bbox=dict(facecolor='0.7',edgecolor='none',pad=3.0))

    xvals = np.linspace(400, 600)
    yvals = np.sqrt(r ** 2 - (xvals - L) ** 2) + L
    yvals[np.isnan(yvals)] = 500
    y2vals = np.ones(len(xvals)) * 600

    ax.fill_between(xvals, yvals, y2vals, alpha=0.75, hatch='//', facecolor='none', edgecolor='k', linewidth=0)
    ax.fill_between(xvals, 2 * L - yvals, np.ones(len(xvals)) * 400, alpha=0.75, hatch='//', facecolor='none',
                    edgecolor='k', linewidth=0)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    cbar = fig.colorbar(a,label=r"$\textrm{Probability}$",location='right',
                 fraction=0.046)
    cbar.set_ticks(ticks=[])
    fig.savefig("/home/fransces/Documents/Figures/Paper/VizColorbarNew.pdf", bbox_inches='tight')

# create t=0, t=1, t=2 visualizations
def modelGraphic(topDir="/home/fransces/Documents/Figures/Paper"):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    cmap = copy.copy(matplotlib.colormaps.get_cmap("viridis"))
    cmap.set_under(color="white")
    cmap.set_bad(color="white")
    maxT = 3
    func = getRandomDistribution('Dirichlet', [1, 1, 1, 1])
    L = 3
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    limits = [L - 3, L + 3]
    fig, ax = plt.subplots(1,3,figsize=(5, 15), dpi=150)
    vmin=1e-5
    for t in range(maxT):
        # biases are left, down, up, right
        occ = updateOccupancy(occ, t, func)
        im = ax[t].imshow(occ/np.max(occ),cmap=cmap, vmax=1, vmin=vmin, aspect='equal')
        # ax[t].imshow(occ,cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal')
        [ax[t].plot([-1,6],[y+.5,y+.5],'k',lw=1) for y in range(-1,6)]
        [ax[t].plot([x+.5,x+.5],[-1,6],'k',lw=1) for x in range(-1,6)]

        ax[t].set_xlim(limits)
        ax[t].set_ylim(limits)
        ax[t].get_xaxis().set_ticks([])
        ax[t].get_yaxis().set_ticks([])
        ax[t].spines['top'].set_visible(False)
        ax[t].spines['right'].set_visible(False)
        ax[t].spines['bottom'].set_visible(False)
        ax[t].spines['left'].set_visible(False)
    # ax[0].annotate("(a)",xy=(0,1), xycoords='axes fraction', xytext=(+0.5,-0.5),
    #              textcoords='offset fontsize', verticalalignment='top',
    #              bbox=dict(facecolor='0.7',edgecolor='none',pad=3.0))
    path = os.path.join(topDir,"modelGraphicFinal.pdf")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

# helper function to map the list of lambda_ext to colors
def colorsForLambda(lambdaList):
    # normalize values of lambda to be between 0 and 1
    logVals = np.log(lambdaList)
    vals = (logVals - np.min(logVals)) / (np.max(logVals) - np.min(logVals))
    # this goes between teal and green i guess?
    colorList = np.array([[l, 1-l, 1] for l in vals])
    if np.any(np.isnan(colorList)):
        colorList[np.isnan(colorList)] = 0
    return colorList


def visualizeLambdaColors():
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
    # pathList = [path003, path01, path03, path1, path3, path10, path31,
    #             pathLogNormal, pathDelta, pathCorner]
    minimalStatsList = [path31, pathCorner, path1, pathDelta, path03, path01, path003]
    expVarX, lambdaList = d.getListOfLambdas(minimalStatsList)
    plt.ion()
    plt.figure()
    colors = colorsForLambda(lambdaList)
    y1, y2 = 1e-11, 5e2
    for i in range(len(lambdaList)):
        plt.vlines(lambdaList[i], y1,  y2, color=colors[i])
    # this isn't great but going to hardcode in list of paths
    lambdaDict = {'alpha31':lambdaList[0], 'corner':lambdaList[1],'alpha1':lambdaList[2],
                  'delta':lambdaList[3], 'alpha.03':lambdaList[4],'alpha.01':lambdaList[5],
                  'alpha.003':lambdaList[6]}
    sortedDict = {}
    for key in sorted(lambdaDict, key=lambdaDict.get):
        sortedDict[key] = lambdaDict[key]
    return sortedDict

# plots var[ln[p]] vs lambda_ext r^2/t^2
def plotMasterCurve(savePath, statsFileList, fullStatFileList, tMaxList, minLambdaExtVals, fullLambdaVal, markers, verticalLine=True,rMin=3):
    """
    plots given lists of var[lnP] as a function of mastercurve f(lambda,r,t) = lambda r^2/t^2
    """
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,10),constrained_layout=True,dpi=300)
    # mastercurve (ax2), want subset of data
    print('starting mastercurve')
    ax2.set_xlim([1e-11, 5e2])
    ax2.set_ylim([1e-11, 5e2])
    ax2.set(adjustable='box',aspect='equal')
    minColors = colorsForLambda(minLambdaExtVals)
    for i in range(len(statsFileList)):
        # load each file
        print(statsFileList[i])
        file = statsFileList[i]
        tempData, label = d.processStats(file)  # label is distribution name
        # for mastercurve (ax2)
        if verticalLine:
            # this is theoretical prediction
            xLoc = minLambdaExtVals[i]
            ax2.loglog([xLoc,xLoc],[xLoc,5e2], color=minColors[i],linestyle='solid', zorder=0.0000000001)
        for j in range(len(tMaxList)):
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= rMin))).flatten()
            scalingFuncVals = d.masterCurveValue(tempData[1, :][indices], tempData[2, :][indices],
                                                 tempData[3, :][indices])
            # for main collapse (vlp vs masterfunc linear)
            ax2.loglog(scalingFuncVals, tempData[0, :][indices],
                       markers[i], color=minColors[i], markeredgecolor='k',
                       ms=4, mew=0.5, label=label, zorder=np.random.rand(), rasterized=True)

    # constant collapse (ax1), want all data
    print("starting constant collapse fig")
    fullColors = colorsForLambda(fullLambdaVal)
    scalingFuncAll, vlpAll, vsAll, timesAll, lsAll = d.prepLossFunc(fullStatFileList, tMaxList,
                                                     vlpMax=1e-3,alpha=1)
    g = vlpAll / (lsAll * vsAll**2)
    print(f"mean g, std g: {np.mean(g), np.std(g)}")
    tedge = np.geomspace(10, 1e4)
    binnedMedianG = [np.median(g[(timesAll > tedge[i]) * (timesAll < tedge[i + 1])]) for i in range(len(tedge) - 1)]
    ax1.semilogx(tedge[1:], binnedMedianG, label="binned median g",color='black',zorder=1000000000000)
    ax1.set_ylim([0,8/3])  # should be a fctor of 2 from the previous one
    ax1.set_xlim([1,2e4])
    ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\displaystyle\frac{ t^2}{r^2 \lambda_{\mathrm{ext}}}\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right)}\right]$")
    for i in range(len(fullStatFileList)):
        file2 = fullStatFileList[i]
        tempData2, label2 = d.processStats(file2)  # label is distribution name
        print(f"{file}")
        # for constant collapse (ax1)
        vlp = tempData2[0,:]  # var[ln[P(r(t))]]
        r = tempData2[1,:]  # radii
        t = tempData2[2,:]  # time
        l = tempData2[3,0]  # lambda_ext
        indices = (r >= rMin)
        # velocities
        vLin = r / t**(1)
        ax1.semilogx(t[indices], (1/(l*vLin[indices]**2))*vlp[indices],'.',color=fullColors[i],
                 alpha=.05,zorder=np.random.rand(), rasterized=True)
        # # binned median for each nu
        # # use vlp < 1e-3 and r<=rMin
        scalingFuncAll, vlpAll, vsAll, timesAll, lsAll = d.prepLossFunc([file2], tMaxList,
                                                                        vlpMax=1e-3, alpha=1)
        g = vlpAll / (lsAll * vsAll ** 2)
        print(f"nu, mean g, std g: {label, np.mean(g), np.std(g)}")
        binnedMedianG = [np.median(g[(timesAll > tedge[i]) * (timesAll < tedge[i + 1])]) for i in range(len(tedge) - 1)]
        # to create a gray line behind each line
        ax1.semilogx(tedge[1:], binnedMedianG, linewidth=1.8, color=[0.1]*3)  # gray
        ax1.semilogx(tedge[1:], binnedMedianG, color=fullColors[i])  # actual line

    # for normal mastercurve (ax2)
    ax2.set_xlabel(r"$\displaystyle\frac{\lambda_{\mathrm{ext}}r^2}{t^2}$")
    ax2.set_ylabel(r"$\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right)}\right]$")
    ax2.set_xticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
    ax2.set_yticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])

    fig.savefig(savePath)

# plots -mean[ln[p]] vs r^2/t
def plotMean(savePath, statsFileList, tMaxList, lambdaExtVals, markers,rMin=3):
    print("starting mean plot")
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    colors = colorsForLambda(lambdaExtVals)
    fig, ax = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=300)
    for i in range(len(statsFileList)):
        print(statsFileList[i])
        for j in range(len(tMaxList)):
            file = statsFileList[i]
            # label: distribution name
            tempData, label = d.processStats(file)
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= rMin))).flatten()

            # mean inset
            with np.errstate(divide='ignore'):
                # x-axis, r^2/ t
                gaussianbehavior = tempData[1, indices]**2 / tempData[2,indices]
                # plot -<lnP> vs r^2/t
                ax.loglog(gaussianbehavior, -tempData[4,:][indices],markers[i],
                           color=colors[i],markeredgecolor='k',ms=4,mew=0.5,label=label,
                           zorder=np.random.rand(), rasterized=True)
    # prediction
    x = np.logspace(-4,3)
    ax.plot(x,x,color='red')
    ax.set(adjustable='box',aspect='equal')
    ax.set_xlabel(r"$r^2 / t$")
    ax.set_ylabel(r"$-\mathbb{E}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right)}\right]$")
    ax.set_xlim([1e-4, 1e3])
    ax.set_ylim([1e-4, 1e3])
    ax.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])
    ax.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])
    fig.savefig(savePath)


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
    fullList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]
    expVarXListFull, lambdaListFull = d.getListOfLambdas(fullList)
    fullMarkers = ['o'] * 7 + ['D'] + ['v'] + ['s']

    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])

    # # for dropping distributions which have very close values of lambda
    minSavePath = "/home/fransces/Documents/Figures/2DRWREMasterCurveReduced.pdf"
    minimalStatsList = [path31, pathCorner, path1, pathDelta, path03, path01, path003]
    minExpVarXList, minLambdaList = d.getListOfLambdas(minimalStatsList)
    minMarkers = ['o'] + ['s'] + ['o'] + ['v'] + ['o']*3
    plotMasterCurve(minSavePath, minimalStatsList, fullList, tMaxList,
                    minLambdaList, lambdaListFull, minMarkers, verticalLine=True)

    # meanPath2 = "/home/fransces/Documents/Figures/2DRWREMeanFull.pdf"
    # plotMean(meanPath2, fullList, tMaxList, markers=fullMarkers, lambdaExtVals=lambdaListFull)