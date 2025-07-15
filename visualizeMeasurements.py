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
#from tqdm import tqdm
from matplotlib.patches import FancyArrowPatch


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


# # not used as much anymore because we switched to get rid of v dependence
# def visualizeAlphaAndVar(savePath, regime='tOnSqrtLogT'):
#     # scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
#     plt.rcParams.update(
#         {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
#     if regime == 'tOnSqrtLogT':  # default to crit regime
#         regimeIdx = 3
#     elif regime == 'sqrt':
#         regimeIdx = 1
#     elif regime == 'linear':
#         regimeIdx = 0
#     os.makedirs(savePath, exist_ok=True)
#     path003 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.03162278/L5000/tMax10000"
#     path01 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.1/L5000/tMax10000"
#     path03 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.31622777/L5000/tMax10000"
#     path1 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000"
#     path3 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000/"
#     path10 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000"
#     path31 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000/"
#     var003 = h5py.File(f"{path003}/Stats.h5", "r")[regime]['var'][:, :]
#     var01 = h5py.File(f"{path01}/Stats.h5", "r")[regime]['var'][:, :]
#     var03 = h5py.File(f"{path03}/Stats.h5", "r")[regime]['var'][:, :]
#     var1 = h5py.File(f"{path1}/Stats.h5", "r")[regime]['var'][:, :]
#     var3 = h5py.File(f"{path3}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array
#     var10 = h5py.File(f"{path10}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array
#     var31 = h5py.File(f"{path31}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array
#
#     # grab list of velocities for big alpha (1e-5 to 10)
#     with open(f"{path10}/variables.json", "r") as v:
#         variables = json.load(v)
#     vels = np.array(variables['velocities'])
#     time = variables['ts']
#     # indexing to get time
#     idx = [-2, -25, -50, -75, -100, -125, -150, -175, -200, -225, -250, -275, -300]
#     x = np.logspace(-10, 0)
#     plt.ion()
#     n = 10
#     colors = plt.cm.jet(np.linspace(0, 1, n))
#     # calculation of Var_nu [E^xi [x]] for dirichlet
#     alphas = [0.03162278, 0.1, 0.31622777, 1, 3.1622776, 10, 31.622776]
#     VarEX = []  # 0.03, 0.1, 0.3, 1, 3, 10, 31
#     for i in range(n - 3):
#         # print(f"alpha: {alphas[i]}")
#         params = np.array([alphas[i]] * 4)
#         VarEX.append(m.getExpVarXDotProduct('Dirichlet', params))
#         # print(f"VarEX: {VarEX}")
#     # now need to do logNormal and randomDelta?
#     pathLogNormal = "data/memoryEfficientMeasurements/h5data/logNormal/0,1/L5000/tMax10000/"
#     varLogNorm = h5py.File(f"{pathLogNormal}/Stats.h5", "r")[regime]['var'][:, :]
#     VarEXLogNorm = m.getExpVarXDotProduct("LogNormal", np.array([0, 1]))
#     pathDelta = "data/memoryEfficientMeasurements/h5data/Delta/L5000/tMax10000/"
#     varDelta = h5py.File(f"{pathDelta}/Stats.h5", "r")[regime]['var'][:, :]
#     VarEXDelta = m.getExpVarXDotProduct("Delta", "")
#     pathCorner = "data/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000"
#     varCorner = h5py.File(f"{pathCorner}/Stats.h5", "r")[regime]['var'][:, :]
#     VarEXCorner = m.getExpVarXDotProduct("Corner", "")  # returns 1/6. analytics gives 1/12.
#     for timeIDX in idx:
#         t = time[timeIDX]
#         # print(f"time: {t}")
#         if t < 100:
#             tStr = f"00{t}"
#         elif 100 <= t < 1000:
#             tStr = f"0{t}"
#         else:
#             tStr = f"{t}"
#         lastVar003 = var003[timeIDX, :]
#         lastVar01 = var01[timeIDX, :]
#         lastVar03 = var03[timeIDX, :]
#         lastVar1 = var1[timeIDX, :]
#         lastVar3 = var3[timeIDX, :]
#         lastVar10 = var10[timeIDX, :]
#         lastVar31 = var31[timeIDX, :]
#         lastVarLogNorm = varLogNorm[timeIDX, :]
#         lastVarDelta = varDelta[timeIDX, :]
#         lastVarCorner = varCorner[timeIDX, :]
#         if regime == 'tOnSqrtLogT':  # default to crit regime
#             prefactor = 1
#             xLabel = r"$v^2 \frac{\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
#             ylabel = r"$\mathrm{Var}[P(\frac{vt}{\sqrt{\ln t}})]$," + f" at t={t}"
#         elif regime == 'sqrt':
#             prefactor = (np.log(t) / t)
#             xLabel = r"$v^2 \frac{\log{t}}{t} \frac{\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
#             ylabel = r"$\mathrm{Var}[P(vt^{1/2})]$" + f" at t={t}"
#         elif regime == 'linear':
#             prefactor = (np.log(t) / (4 * np.pi))
#             xLabel = r"$v^2 \log{t} \frac{\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
#             ylabel = r"$\mathrm{Var}[P(vt)]$" + f" at t={t}"
#         plt.figure(figsize=(5, 4), constrained_layout=True, dpi=150)
#         plt.loglog(prefactor * (VarEX[0] / (1 - VarEX[0])) * vels ** 2, lastVar003, '.', color=colors[0],
#                    label=r"$\alpha= 0.03$")
#         plt.loglog(prefactor * (VarEX[1] / (1 - VarEX[1])) * vels ** 2, lastVar01, '.', color=colors[1],
#                    label=r"$\alpha= 0.1$")
#         plt.loglog(prefactor * (VarEX[2] / (1 - VarEX[2])) * vels ** 2, lastVar03, '.', color=colors[2],
#                    label=r"$\alpha= 0.3$")
#         plt.loglog(prefactor * (VarEX[3] / (1 - VarEX[3])) * vels ** 2, lastVar1, '.', color=colors[3],
#                    label=r"$\alpha= 1$")
#         plt.loglog(prefactor * (VarEX[4] / (1 - VarEX[4])) * vels ** 2, lastVar3, '.', color=colors[4],
#                    label=r"$\alpha= 3$")
#         plt.loglog(prefactor * (VarEX[5] / (1 - VarEX[5])) * vels ** 2, lastVar10, '.', color=colors[5],
#                    label=r"$\alpha= 10$")
#         plt.loglog(prefactor * (VarEX[6] / (1 - VarEX[6])) * vels ** 2, lastVar31, '.', color=colors[6],
#                    label=r"$\alpha= 31$")
#         plt.loglog(prefactor * (VarEXLogNorm / (1 - VarEXLogNorm)) * vels ** 2, lastVarLogNorm, '.', color=colors[7],
#                    label=r"LogNormal(0,1)")
#         plt.loglog(prefactor * (VarEXDelta / (1 - VarEXDelta)) * vels ** 2, lastVarDelta, '.', color=colors[8],
#                    label="Delta")
#         plt.loglog((prefactor * (VarEXCorner / (1 - VarEXCorner)) * vels ** 2), lastVarCorner, '*', color=colors[9],
#                    label="Corner")
#         plt.plot((4 * np.pi) * x, x, color='k', linestyle='dashed', label=r"y=4pi x")
#         plt.ylim([10 ** -8, 10 ** 3])
#         plt.xlim([10 ** -8, 10 ** 3])
#         plt.xlabel(xLabel)
#         plt.ylabel(ylabel)
#         # plt.legend(loc=2)
#         plt.savefig(f"{savePath}/" + tStr + ".pdf")
#

def measurementPastCircleGraphic():
    plt.rcParams.update({'font.size': 20, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})

    maxT = 500
    func = getRandomDistribution('Dirichlet', [1, 1, 1, 1])

    L = 500
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1

    for t in range(maxT):
        occ = updateOccupancy(occ, t, func)

    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under(color="white")
    cmap.set_bad(color="white")
    vmax = np.max(occ)
    vmin = 1e-10

    fig, ax = plt.subplots()
    ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, interpolation='none',alpha=1)
    limits = [L - 100, L + 100]
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    r = 75
    circle = plt.Circle((L, L), r, color='k', ls='--', fill=False, lw=2.5)
    ax.add_patch(circle)
    arrow = FancyArrowPatch((L, L), (L + r * np.cos(np.pi / 4), L + r * np.sin(np.pi / 4)), color='k',
                            mutation_scale=40)
    ax.add_patch(arrow)

    ax.annotate(r"$r(t)$", (500, 545))

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

    fig.savefig("/home/fransces/Documents/Figures/Viz.pdf", bbox_inches='tight')

def modelGraphic(topDir="/home/fransces/Documents/Figures/Paper"):
    plt.rcParams.update({'font.size': 20, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})
    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under(color="white")
    cmap.set_bad(color="white")

    maxT = 3
    func = getRandomDistribution('Dirichlet', [1, 1, 1, 1])

    L = 3
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    limits = [L - 3, L + 3]
    fig, ax = plt.subplots(1,3,figsize=(5, 15), dpi=150)
    for t in range(maxT):
        occ = updateOccupancy(occ, t, func)
        vmax = np.max(occ)
        vmin = 1e-10
        ax[t].imshow(occ,cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal')
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
    path = os.path.join(topDir,"modelGraphic.pdf")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def colorsForLambda(lambdaList):
    # normalize values of lambda to be between 0 and 1
    logVals = np.log(lambdaList)
    vals = (logVals - np.min(logVals)) / (np.max(logVals) - np.min(logVals))
    # this goes between teal and green i guess?
    colorList = np.array([[l, 1-l, 1] for l in vals])
    if np.any(np.isnan(colorList)):
        colorList[np.isnan(colorList)] = 0
    return colorList


def plotMasterCurve(savePath, statsFileList, tMaxList, lambdaExtVals, markers, verticalLine=True):
    """
    plots given lists of var[lnP] as a function of mastercurve f(lambda,r,t) = lambda r^2/t^2
    """
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    plt.ion()
    colors = colorsForLambda(lambdaExtVals)
    plt.figure(figsize=(5, 5), constrained_layout=True, dpi=150)
    plt.xlim([1e-11, 5e2])
    plt.ylim([1e-11, 5e2])
    # plt.xlim([1e0,1e4])
    # plt.ylim([1e-1,1e1])
    plt.gca().set_aspect('equal')
    # TODO: make more robust? under the assumption that all paths have the same ts list.
    for i in range(len(statsFileList)):
        print(statsFileList[i])
        if verticalLine:
            # this part is data for alpha=1
            # pathDiamondVar = "/home/fransces/Documents/code/diamondVars.npy"
            # diamondVals = np.load(pathDiamondVar)
            # varMasterCurves = d.masterCurveValue(diamondVals[0, :], diamondVals[0, :], lambdaExtVals[1])
            # plt.plot(varMasterCurves, diamondVals[1, :], color='red')

            # this is theoretical prediction
            xLoc = lambdaExtVals[i]
            plt.loglog([xLoc,xLoc],[xLoc,5e2], color=colors[i],linestyle='solid', zorder=0.0000000001)
        for j in range(len(tMaxList)):
            file = statsFileList[i]
            # label: distribution name
            tempData, label = d.processStats(file)
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= 2))).flatten()
            scalingFuncVals = d.masterCurveValue(tempData[1, :][indices], tempData[2, :][indices],
                                                 tempData[3, :][indices])
            # for main collapse (vlp vs masterfunc linear)
            plt.loglog(scalingFuncVals, tempData[0, :][indices],
                       markers[i], color=colors[i], markeredgecolor='k',
                       ms=4, mew=0.5, label=label, zorder=np.random.rand())
    # for normal mastercurve
    plt.xlabel(r"$\frac{\displaystyle\lambda_{\mathrm{ext}}r^2}{t^2}$")
    plt.ylabel(r"$\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(r\right)\right)}\right]$")
    # # for vlp/mastercurve vs time
    # plt.xlabel("t")
    # plt.ylabel("vlp / f(r,t,lambda)")

    # offset?
    x = np.logspace(-8, -5)
    plt.plot(x, x+1e-11, color='k',linestyle='dashed')

    plt.xticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
    plt.yticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
    plt.savefig(f"{savePath}")


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
    markers = ['o'] * 7 + ['D'] + ['v'] + ['s']
    savePath = "/home/fransces/Documents/Figures/Paper/2DRWREMasterCurveFinal.pdf"

    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])

    # # this will run through all your files once to pull out the
    # # list of lambdas. the order should correspond to the filelist
    # expVarXList, lambdaList = d.getListOfLambdas(statsFileList)
    # # this is the second runthrough (all data, all times)
    # plotMasterCurve(savePath, statsFileList, tMaxList, markers=markers,lambdaExtVals=lambdaList,verticalLine=True)
    # # # for only dirichlet alpha=1 all times
    # for all data of one value of alpha
    # singleFile = [path1]
    # savePathSingle = "/home/fransces/Documents/Figures/testfigs/2DRWREMasterCurveDirichlet1.png"
    # # plotMasterCurve(savePathSingle, singleFile, tMaxList=tMaxList, lambdaExtVals=[lambdaList[3]]*3,markers=markers)

    # for dropping distributions which have very close values of lambda
    minSavePath = "/home/fransces/Documents/Figures/testfigs/2DRWREMasterCurveReduced.pdf"
    minimalStatsList = [path31, pathCorner, path1, pathDelta, path03, path01, path003]
    minExpVarXList, minLambdaList = d.getListOfLambdas(minimalStatsList)
    minmarkers = ['o'] + ['s'] + ['o'] + ['v'] + ['o']*3
    plotMasterCurve(minSavePath, minimalStatsList, tMaxList, markers=minmarkers, lambdaExtVals=minLambdaList, verticalLine=True)