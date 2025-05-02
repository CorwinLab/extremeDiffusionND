import memEfficientEvolve2DLattice as m
import os
import h5py
import json
import matplotlib.patches
import numpy as np
from matplotlib import pyplot as plt
from memEfficientEvolve2DLattice import evolve2DDirichlet
import matplotlib
from randNumberGeneration import getRandomDistribution
import dataAnalysis as d
import sys


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


# not used as much anymore because we switched to get rid of v dependence
def visualizeAlphaAndVar(savePath, regime='tOnSqrtLogT'):
    # scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    if regime == 'tOnSqrtLogT':  # default to crit regime
        regimeIdx = 3
    elif regime == 'sqrt':
        regimeIdx = 1
    elif regime == 'linear':
        regimeIdx = 0
    os.makedirs(savePath, exist_ok=True)
    path003 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.03162278/L5000/tMax10000"
    path01 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.1/L5000/tMax10000"
    path03 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.31622777/L5000/tMax10000"
    path1 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000"
    path3 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000/"
    path10 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000"
    path31 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000/"
    var003 = h5py.File(f"{path003}/Stats.h5", "r")[regime]['var'][:, :]
    var01 = h5py.File(f"{path01}/Stats.h5", "r")[regime]['var'][:, :]
    var03 = h5py.File(f"{path03}/Stats.h5", "r")[regime]['var'][:, :]
    var1 = h5py.File(f"{path1}/Stats.h5", "r")[regime]['var'][:, :]
    var3 = h5py.File(f"{path3}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array
    var10 = h5py.File(f"{path10}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array
    var31 = h5py.File(f"{path31}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array

    # grab list of velocities for big alpha (1e-5 to 10)
    with open(f"{path10}/variables.json", "r") as v:
        variables = json.load(v)
    vels = np.array(variables['velocities'])
    time = variables['ts']
    # indexing to get time
    idx = [-2, -25, -50, -75, -100, -125, -150, -175, -200, -225, -250, -275, -300]
    x = np.logspace(-10, 0)
    plt.ion()
    n = 10
    colors = plt.cm.jet(np.linspace(0, 1, n))
    # calculation of Var_nu [E^xi [x]] for dirichlet
    alphas = [0.03162278, 0.1, 0.31622777, 1, 3.1622776, 10, 31.622776]
    VarEX = []  # 0.03, 0.1, 0.3, 1, 3, 10, 31
    for i in range(n - 3):
        # print(f"alpha: {alphas[i]}")
        params = np.array([alphas[i]] * 4)
        VarEX.append(m.getExpVarX('Dirichlet', params))
        # print(f"VarEX: {VarEX}")
    # now need to do logNormal and randomDelta?
    pathLogNormal = "data/memoryEfficientMeasurements/h5data/logNormal/0,1/L5000/tMax10000/"
    varLogNorm = h5py.File(f"{pathLogNormal}/Stats.h5", "r")[regime]['var'][:, :]
    VarEXLogNorm = m.getExpVarX("LogNormal", np.array([0, 1]))
    pathDelta = "data/memoryEfficientMeasurements/h5data/Delta/L5000/tMax10000/"
    varDelta = h5py.File(f"{pathDelta}/Stats.h5", "r")[regime]['var'][:, :]
    VarEXDelta = m.getExpVarX("Delta", "")
    pathCorner = "data/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000"
    varCorner = h5py.File(f"{pathCorner}/Stats.h5", "r")[regime]['var'][:, :]
    VarEXCorner = m.getExpVarXDotProduct("Corner", "")  # returns 1/6. analytics gives 1/12.
    for timeIDX in idx:
        t = time[timeIDX]
        # print(f"time: {t}")
        if t < 100:
            tStr = f"00{t}"
        elif 100 <= t < 1000:
            tStr = f"0{t}"
        else:
            tStr = f"{t}"
        lastVar003 = var003[timeIDX, :]
        lastVar01 = var01[timeIDX, :]
        lastVar03 = var03[timeIDX, :]
        lastVar1 = var1[timeIDX, :]
        lastVar3 = var3[timeIDX, :]
        lastVar10 = var10[timeIDX, :]
        lastVar31 = var31[timeIDX, :]
        lastVarLogNorm = varLogNorm[timeIDX, :]
        lastVarDelta = varDelta[timeIDX, :]
        lastVarCorner = varCorner[timeIDX, :]
        if regime == 'tOnSqrtLogT':  # default to crit regime
            prefactor = 1
            xLabel = r"$v^2 \frac{\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
            ylabel = r"$\mathrm{Var}[P(\frac{vt}{\sqrt{\ln t}})]$," + f" at t={t}"
        elif regime == 'sqrt':
            prefactor = (np.log(t) / t)
            xLabel = r"$v^2 \frac{\log{t}}{t} \frac{\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
            ylabel = r"$\mathrm{Var}[P(vt^{1/2})]$" + f" at t={t}"
        elif regime == 'linear':
            prefactor = (np.log(t) / (4 * np.pi))
            xLabel = r"$v^2 \log{t} \frac{\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-\mathrm{Var}_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
            ylabel = r"$\mathrm{Var}[P(vt)]$" + f" at t={t}"
        plt.figure(figsize=(5, 4), constrained_layout=True, dpi=150)
        plt.loglog(prefactor * (VarEX[0] / (1 - VarEX[0])) * vels ** 2, lastVar003, '.', color=colors[0],
                   label=r"$\alpha= 0.03$")
        plt.loglog(prefactor * (VarEX[1] / (1 - VarEX[1])) * vels ** 2, lastVar01, '.', color=colors[1],
                   label=r"$\alpha= 0.1$")
        plt.loglog(prefactor * (VarEX[2] / (1 - VarEX[2])) * vels ** 2, lastVar03, '.', color=colors[2],
                   label=r"$\alpha= 0.3$")
        plt.loglog(prefactor * (VarEX[3] / (1 - VarEX[3])) * vels ** 2, lastVar1, '.', color=colors[3],
                   label=r"$\alpha= 1$")
        plt.loglog(prefactor * (VarEX[4] / (1 - VarEX[4])) * vels ** 2, lastVar3, '.', color=colors[4],
                   label=r"$\alpha= 3$")
        plt.loglog(prefactor * (VarEX[5] / (1 - VarEX[5])) * vels ** 2, lastVar10, '.', color=colors[5],
                   label=r"$\alpha= 10$")
        plt.loglog(prefactor * (VarEX[6] / (1 - VarEX[6])) * vels ** 2, lastVar31, '.', color=colors[6],
                   label=r"$\alpha= 31$")
        plt.loglog(prefactor * (VarEXLogNorm / (1 - VarEXLogNorm)) * vels ** 2, lastVarLogNorm, '.', color=colors[7],
                   label=r"LogNormal(0,1)")
        plt.loglog(prefactor * (VarEXDelta / (1 - VarEXDelta)) * vels ** 2, lastVarDelta, '.', color=colors[8],
                   label="Delta")
        plt.loglog((prefactor * (VarEXCorner / (1 - VarEXCorner)) * vels ** 2), lastVarCorner, '*', color=colors[9],
                   label="Corner")
        plt.plot((4 * np.pi) * x, x, color='k', linestyle='dashed', label=r"y=4pi x")
        plt.ylim([10 ** -8, 10 ** 3])
        plt.xlim([10 ** -8, 10 ** 3])
        plt.xlabel(xLabel)
        plt.ylabel(ylabel)
        # plt.legend(loc=2)
        plt.savefig(f"{savePath}/" + tStr + ".png")


def createMeasurementGraphic(saveFile):
    """
    savefile: str, path to which fig is saved
    """
    # os.makedirs(saveFile,exist_ok=True)
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    maxT = 500
    distName = 'Dirichlet'
    params = np.array([1, 1, 1, 1])
    pdf = False
    occtype = int
    occupancy = np.zeros(shape=(2 * maxT + 1, 2 * maxT + 1))
    occupancy[maxT, maxT] = 1
    # v = 0.63

    func = getRandomDistribution(distName, params)

    for t, occ in evolve2DDirichlet(occupancy, maxT, func):
        pass

    cmap = matplotlib.colormaps["jet"]  # instead of get_cmap because newer matplotlib version
    cmap.set_under(color="white")
    cmap.set_bad(color="white")
    vmax = np.max(occ)
    vmin = 1e-9

    r1 = 25
    r2 = 60
    r3 = 3.5 * np.sqrt(maxT)

    # # with velocities
    # r1 = v * np.sqrt(maxT)
    # r2 = (v * maxT) / (np.sqrt(np.log(maxT)))
    # r3 = v * maxT

    circle1 = plt.Circle((maxT, maxT), r1, color='k', fill=False, ls='--')
    circle2 = plt.Circle((maxT, maxT), r2, color='k', fill=False, ls='--')
    circle3 = plt.Circle((maxT, maxT), r3, color='k', fill=False, ls='--')

    mutation_scale = 10

    arrow1 = matplotlib.patches.FancyArrowPatch((maxT, maxT),
                                                (maxT + r1 * np.cos(-np.pi / 3), maxT + r1 * np.sin(-np.pi / 3)),
                                                mutation_scale=mutation_scale, color='k')
    arrow2 = matplotlib.patches.FancyArrowPatch((maxT, maxT),
                                                (maxT + r2 * np.cos(np.pi / 6), maxT + r2 * np.sin(np.pi / 6)),
                                                mutation_scale=mutation_scale, color='k')
    arrow3 = matplotlib.patches.FancyArrowPatch((maxT, maxT), (maxT, maxT + r3), mutation_scale=mutation_scale,
                                                color='k')

    fig, ax = plt.subplots()
    width = 4 * np.sqrt(maxT)
    # if using pre-existing occ then use L instead of maxT
    ax.set_xlim([maxT - width, maxT + width])
    ax.set_ylim([maxT - width, maxT + width])

    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)

    ax.add_patch(arrow1)
    ax.add_patch(arrow2)
    ax.add_patch(arrow3)

    ax.annotate(r"$v t$", (505, 562.5))
    ax.annotate(r"$v \sqrt{t}$", (490, 480))
    ax.annotate(r"$v \frac{t}{\sqrt{\ln(t)}}$", (525, 508))
    ax.set_axis_off()

    # setting norm=matplotlib.colors.LogNorm automatically takes the log (base 10)
    # trying to do like np.log(occ) and not do the norm gets back the terrible checkerboard
    # ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
    #         cmap=cmap,  interpolation='gaussian', alpha=0.75)
    ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
              cmap=cmap, alpha=0.75)
    fig.savefig(f"{saveFile}.png", bbox_inches='tight')


# TODO: fix labels acc. to eric's message on 12 feb 2025
def plotAllSystems(path, saveDir, regime='tOnSqrtLogT', takeLog=False):
    files = os.listdir(path)
    os.makedirs(saveDir, exist_ok=True)
    files.remove('info.npz')
    files.remove('statsNoLog.npz')
    files.remove('stats.npz')
    info = np.load(f"{path}/info.npz")
    velocities = info['velocities'].flatten()
    vIdx = -8  # hard code in 0.39 for v, for figure purposes
    time = info['times']
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    if regime == 'tOnSqrtLogT':  # default to crit regime
        regimeIdx = 3
        r = velocities[vIdx] * time / (np.sqrt(np.log(time)))
        if takeLog:
            ylabel = r"$-\ln{P(\frac{vt}{\sqrt{\ln t}})}$"
            meanLabel = r"-$\langle \ln{P(\frac{vt}{\sqrt{\ln t}})} \rangle$"
        else:
            ylabel = r"$P(\frac{vt}{\sqrt{\ln t}})$"
            meanLabel = r"$\langle P(\frac{vt}{\sqrt{\ln t}}) \rangle$"
    elif regime == 'sqrt':
        regimeIdx = 1
        r = velocities[vIdx] * (time ** (1 / 2))
        if takeLog:
            ylabel = r"$-\ln{P(vt^{1/2})}$"
            meanLabel = r"$-\langle \ln{P(vt^{1/2})} \rangle$"
        else:
            ylabel = r"$ P(vt^{1/2}) $"
            meanLabel = r"$\langle P(vt^{1/2}) \rangle$"
    elif regime == 'linear':
        regimeIdx = 0
        r = velocities[vIdx] * time
        if takeLog:
            ylabel = r"$-\ln{P(vt)}"
            meanLabel = r"$-\langle \ln{P(vt)} \rangle$"
        else:
            ylabel = r"$P(vt)"
            meanLabel = r"$\langle P(vt) \rangle$"
    if takeLog:
        statsMean = np.load(f"{path}/stats.npz")['mean'][regimeIdx, :, vIdx]
    else:
        statsMean = np.load(f"{path}/statsNoLog.npz")['mean'][regimeIdx, :, vIdx]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150, constrained_layout=True)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    for file in files:
        temp = np.load(f"{path}/{file}")[regimeIdx, :, vIdx]
        #TODO: what if dots plus lines instead
        if takeLog:
            with np.errstate(divide='ignore'):
                ax.loglog(time, -np.log(temp), '.', linewidth=1, alpha=0.4)
        else:
            ax.semilogx(time, temp, '.', linewidth=1, alpha=0.4)
    if takeLog:
        with np.errstate(divide='ignore'):
            ax.loglog(time, - statsMean, 'o', color='k', alpha=0.8, label=meanLabel)
            prediction = r ** 2 / time
            ax.loglog(time, prediction, color='red', linestyle='dashed', label=r"$\frac{r(t)^2}{t}$")
        savePath = f"{saveDir}/{regime}" + "AllSystemsLnP.png"
    else:
        ax.semilogx(time, statsMean, 'o', color='k', label=meanLabel)
        savePath = f"{saveDir}/{regime}" + "AllSystemsRawP.png"
    ax.legend()
    #ax.set_ylim([1e-6,1])
    fig.savefig(savePath, bbox_inches='tight')


def plotAllVariance(path, saveDir, regime='tOnSqrtLogT'):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    os.makedirs(saveDir, exist_ok=True)
    info = np.load(f"{path}/info.npz")
    velocities = info['velocities'].flatten()
    time = info['times']
    VarEx = m.getExpVarX("Dirichlet", [0.1] * 4)
    if regime == 'tOnSqrtLogT':  # default to crit regime
        regimeIdx = 3
        ylabel = r"$\frac{1-\mathrm{Var}_{\nu}[\mathbb{E}^{\xi}[\vec{X}]]}{v^2 \mathrm{Var}_{\nu}[\mathbb{E}^{\xi}[\vec{X}]]} \mathrm{Var}[\ln{P(\frac{vt}{\sqrt{\ln t}})}]$"
    elif regime == 'sqrt':
        regimeIdx = 1
        ylabel = r"$\frac{1-\mathrm{Var}_{\nu}[\mathbb{E}^{\xi}[\vec{X}]]}{v^2 \mathrm{Var}_{\nu}[\mathbb{E}^{\xi}[\vec{X}]]} \mathrm{Var}[\ln{P(vt^{1/2})}]$"
    elif regime == 'linear':
        regimeIdx = 0
        ylabel = r"$\frac{1-\mathrm{Var}_{\nu}[\mathbb{E}^{\xi}[\vec{X}]]}{v^2 \mathrm{Var}_{\nu}[\mathbb{E}^{\xi}[\vec{X}]]} \mathrm{Var}[\ln{P(vt)}]$"
    statsVar = np.load(f"{path}/stats.npz")['variance'][regimeIdx, :, :]
    n = statsVar.shape[1]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150, constrained_layout=True)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    colors = plt.cm.jet(np.linspace(0, 1, n))
    for i in range(n):
        with np.errstate(divide='ignore'):
            scaledPrefactor = 1 / ((1 / (4 * np.pi)) * velocities[i] ** 2 * (VarEx / (1 - VarEx)))
            ax.loglog(time[velocities[i] * time / (np.sqrt(np.log(time))) > 2],
                      scaledPrefactor * statsVar[:, i][velocities[i] * time / (np.sqrt(np.log(time))) > 2], '.',
                      color=colors[i], alpha=0.8, label=np.round(velocities[i], 2))
            savePath = f"{saveDir}/{regime}" + "VarLnP.png"
    fig.savefig(savePath, bbox_inches='tight')


def colorsForLambda(lambdaList):
    # normalize values of lambda to be between 0 and 1
    logVals = np.log(lambdaList)
    #vals = (lambdaList - np.min(lambdaList)) / (np.max(lambdaList) - np.min(lambdaList))
    vals = (logVals - np.min(logVals)) / (np.max(logVals) - np.min(logVals))
    #print(f"vals: {vals}")
    # this goes between teal and green i guess?
    colorList = [[l, 1-l, 1] for l in vals]
    return colorList


def plotMasterCurve(savePath, statsFileList, tMaxList, lambdaExtVals):
    """

    Parameters
    ----------
    savePath

    Returns
    -------

    """
    # os.makedirs(savePath, exist_ok=True)
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    plt.ion()
    # n = len(statsFileList)
    markers = ['o'] * 7 + ['D'] + ['v'] + ['s']
    # correspond transparency to time
    alphaVals = np.flip(np.linspace(0.3, 1, tMaxList.shape[0]))
    #TODO: change colors (single-scale) to agree w/ value of lambda-ext
    # TODO: diff. markers that are the same color
    # then the message is look at everything that's dark brown
    # similar lambda but diff. nu
    # colors = plt.cm.cool(np.linspace(0, 1, n))
    colors = colorsForLambda(lambdaExtVals)
    plt.figure(figsize=(5, 5), constrained_layout=True, dpi=150)
    plt.xlim([1e-11, 5e2])
    plt.ylim([1e-11, 5e2])
    plt.gca().set_aspect('equal')
    # TODO: make more robust? under the assumption that all paths have the same ts list.
    for i in range(len(statsFileList)):
        for j in range(len(tMaxList)):
            print(f"time: {tMaxList[j]}, opacity: {alphaVals[j]}")
            file = statsFileList[i]
            # label: distribution name
            tempData, label = d.processStats(file)
            # we want to get all the values at t-> inf
            # but we ALSO want to mask out the small radii (r<1) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= 2))).flatten()
            scalingFuncVals = d.masterCurveValue(tempData[1, :][indices], tempData[2, :][indices],
                                                 tempData[3, :][indices])
            plt.loglog(scalingFuncVals, tempData[0, :][indices],
                       markers[i], color=colors[i], markeredgecolor='k',
                       ms=4, mew=0.5, label=label, alpha=alphaVals[j], zorder=np.random.rand())
    # plt.legend()
    #plt.xlabel(r"$f\left(r(t),t,\lambda_{\mathrm{ext}}\right)$")
    plt.xlabel(r"$\frac{\lambda_{\mathrm{ext}}}{4\pi}\frac{\ln{t}}{t^2} r^2$")
    plt.ylabel(r"$\mathrm{Var}[\ln{P\left(r\right)}]$")
    x = np.logspace(-11, 2)
    plt.plot(x, x, color='k', linestyle='dashed', label=r"y=4pi x")
    # TODO: figure out how to make aspect ratio equal since slope should be 1
    print('adjusting ticks')
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

    statsFileList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]

    savePath = "/home/fransces/Documents/Figures/2DRWREMasterCurve.png"

    # TODO: add all data (ie at all times)
    # TODO: change from manual list of 2 per decade
    tMaxList = np.flip(np.array([2, 10, 37, 100, 402, 639, 4047, 9816]))

    # this will run through all your files once to pull out the
    # list of lambdas. the order should correspond to the filelist
    expVarXList, lambdaList = d.getListOfLambdas(statsFileList)
    # print(f"exp var X List: {expVarXList}")
    # print(f"lambda List: {lambdaList}")

    # this is the second runthrough
    # its kind of
    plotMasterCurve(savePath, statsFileList, tMaxList=tMaxList, lambdaExtVals=lambdaList)


