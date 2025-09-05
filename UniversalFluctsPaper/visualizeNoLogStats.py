import json
from matplotlib import pyplot as plt
import matplotlib
import dataAnalysis as d
import numpy as np
import matplotlib
from matplotlib.patches import FancyArrowPatch

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

# TODO: turn this into plots of varP vs t and scaled varP vs lambda r^2/t^2
def plotVarP(savePath, statsFileList, fullStatFileList, tMaxList, minLambdaExtVals, fullLambdaVal, markers,
                    verticalLine=True, rMin=3):
    """
    plots given lists of var[lnP] as a function of mastercurve f(lambda,r,t) = lambda r^2/t^2
    """
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,10),constrained_layout=True,dpi=300)
    # mastercurve (ax2), want subset of data
    print('starting mastercurve')
    #ax2.set_xlim([1e-11, 5e2])
    #ax2.set_ylim([1e-11, 5e2])
    # ax2.set(adjustable='box',aspect='equal')
    minColors = colorsForLambda(minLambdaExtVals)
    for i in range(len(statsFileList)):
        # load each file
        print(statsFileList[i])
        file = statsFileList[i]
        # not dependnet on if var[lnP] or varP
        tempData, label = d.processStats(file)  # label is distribution name
        # for mastercurve (ax2)
        if verticalLine:
            # this is theoretical prediction
            xLoc = minLambdaExtVals[i]
            ax2.loglog([xLoc,xLoc],[0,5e2], color=minColors[i],linestyle='solid', zorder=0.0000000001)
        for j in range(len(tMaxList)):
            # grab times we're interested in, and mask out the small radii (r<rmin) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= rMin))).flatten()
            # scaling func  = lambda r^2/t^2, indpt of if var[lnP] or varP
            scalingFuncVals = d.masterCurveValue(tempData[1, :][indices], tempData[2, :][indices],
                                                 tempData[3, :][indices])
            # for main collapse (var vs masterfunc linear)
            ax2.loglog(scalingFuncVals, tempData[0, :][indices],
                       markers[i], color=minColors[i], markeredgecolor='k',
                       ms=4, mew=0.5, label=label, zorder=np.random.rand(), rasterized=True)

    # constant collapse (ax1), want all data
    print("starting constant collapse fig")
    fullColors = colorsForLambda(fullLambdaVal)
    scalingFuncAll, varsAll, vsAll, timesAll, lsAll = d.prepLossFunc(fullStatFileList, tMaxList,
                                                     vlpMax=1e-3,alpha=1)
    g = varsAll / (lsAll * vsAll**2)
    print(f"mean g, std g: {np.mean(g), np.std(g)}")
    tedge = np.geomspace(10, 1e4)
    binnedMedianG = [np.median(g[(timesAll > tedge[i]) * (timesAll < tedge[i + 1])]) for i in range(len(tedge) - 1)]
    ax1.semilogx(tedge[1:], binnedMedianG, label="binned median g",color='black',zorder=1000000000000)
    ax1.set_ylim([1e-7,10])  # should be a fctor of 2 from the previous one
    # ax1.set_xlim([1,2e4])
    # ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\displaystyle\frac{\ln{t} t^2}{r^2 \lambda_{\mathrm{ext}}}\mathrm{Var}_\nu \left[\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right]$")
    for i in range(len(fullStatFileList)):
        file2 = fullStatFileList[i]
        # processStats also not dependent on if varLnP vs varP
        tempData2, label2 = d.processStats(file2)  # label is distribution name
        print(f"{file}")
        # for constant collapse (ax1)
        var = tempData2[0,:]  # var[ln[P(r(t))]]
        r = tempData2[1,:]  # radii
        t = tempData2[2,:]  # time
        l = tempData2[3,0]  # lambda_ext
        indices = (r >= rMin)
        # velocities
        vLin = ( r / t**(1) ) * np.sqrt(np.log(t))
        ax1.loglog(t[indices], (1/(l*vLin[indices]**2))*var[indices],'.',color=fullColors[i],
                 alpha=.05,zorder=np.random.rand(), rasterized=True)
        # # binned median for each nu
        # # use vlp < 1e-3 and r<=rMin
        scalingFuncAll, varsAll, vsAll, timesAll, lsAll = d.prepLossFunc([file2], tMaxList,
                                                                        vlpMax=1e-3, alpha=1)
        g = varsAll / (lsAll * vsAll ** 2)
        print(f"nu, mean g, std g: {label, np.mean(g), np.std(g)}")
        binnedMedianG = [np.median(g[(timesAll > tedge[i]) * (timesAll < tedge[i + 1])]) for i in range(len(tedge) - 1)]
        # to create a gray line behind each line
        ax1.loglog(tedge[1:], binnedMedianG, linewidth=1.8, color=[0.1]*3)  # gray
        ax1.loglog(tedge[1:], binnedMedianG, color=fullColors[i])  # actual line

    # for normal mastercurve (ax2)
    ax2.set_xlabel(r"$\displaystyle\frac{\lambda_{\mathrm{ext}}r^2}{t^2}$")
    ax2.set_ylabel(r"$\mathrm{Var}_\nu \left[\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right]$")
    # ax2.set_xticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
    # ax2.set_yticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])

    fig.savefig(savePath)

# TODO: depends on varLnP vs varP
# plots -mean[ln[p]] vs r^2/t
def plotMeanP(savePath, statsFileList, tMaxList, lambdaExtVals, markers,rMin=3):
    """
    plots mean[P] vs r^2 /  t
        savePath: path to which fig is saved
        statsFileList: list of stats files which will be plotted
        tMaxList: list of all times u want pltoted
        lambdaExtVals: list of all lambda_ext vals from statsFileLIst
        markers: list of markers dependent on orer of statsFileList
        rMin: default 3, the minimum radius from origin which will be considered
    """
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
            tempData, label = d.processStats(file)  # not dependent on varLnP vs varP
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= rMin))).flatten()
            # mean theory
            with np.errstate(divide='ignore'):
                # x-axis, e^( - r^2/t )
                gaussianbehavior = np.exp(- tempData[1, indices]**2 / tempData[2,indices])
                # plot <P> vs e^( - r^2/t)
                ax.loglog(gaussianbehavior, tempData[4,:][indices],markers[i],
                           color=colors[i],markeredgecolor='k',ms=4,mew=0.5,label=label,
                           zorder=np.random.rand(), rasterized=True)
    # prediction
    x = np.logspace(-266,3)
    ax.plot(x,x,color='red')
    ax.set(adjustable='box',aspect='equal')
    ax.set_xlabel(r"$e^{-r^2 / t}$")
    ax.set_ylabel(r"$\mathbb{E}_\nu \left[\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r\right)\right]$")
    # ax.set_xlim([1e-4, 1e3])
    # ax.set_ylim([1e-4, 1e3])
    # ax.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])
    # ax.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])
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

    # for all data, all times
    fullList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]
    expVarXListFull, lambdaListFull = d.getListOfLambdas(fullList)
    fullMarkers = ['o'] * 7 + ['D'] + ['v'] + ['s']

    with open("/mnt/locustData/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])

    figPath = "/home/fransces/Documents/Figures/2DRWREnoLogMasterCurveReduced.pdf"
    # need to use full data for binned median calc, but reduced data for the mastercurve
    minimalStatsList = [path31, pathCorner, path1, pathDelta, path03, path01, path003]
    minExpVarXList, minLambdaList = d.getListOfLambdas(minimalStatsList)
    minMarkers = ['o'] + ['s'] + ['o'] + ['v'] + ['o']*3
    plotVarP(figPath, minimalStatsList, fullList, tMaxList,
                    minLambdaList, lambdaListFull, minMarkers, verticalLine=True)

    # # use all data for mean
    # meanPPath = "/home/fransces/Documents/Figures/2DRWREnoLogMean.pdf"
    # plotMeanP(meanPPath, fullList, tMaxList, markers=fullMarkers, lambdaExtVals=lambdaListFull)