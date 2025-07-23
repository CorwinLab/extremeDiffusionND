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
    plt.rcParams.update({'font.size': 20, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{bm}'})

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
    # to fix colorization of model graphic ? make them all use the same min and max.
    vmax = 1
    # vmax = np.max(occ)
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

    ax.annotate(r"$r(t)$", (500, 545))
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
    fig.colorbar(a,label=r"$\textrm{Probability}$",location='right',
                 fraction=0.046,pad=0.04)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    fig.savefig("/home/fransces/Documents/Figures/Paper/VizColorbarNew.pdf", bbox_inches='tight')


def modelGraphic(topDir="/home/fransces/Documents/Figures/Paper"):
    plt.rcParams.update({'font.size': 20, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})
    cmap = copy.copy(matplotlib.colormaps.get_cmap("viridis"))
    cmap.set_under(color="white")
    cmap.set_bad(color="white")
    maxT = 3
    func = getRandomDistribution('Dirichlet', [1, 1, 1, 1])
    L = 3
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    limits = [L - 3, L + 3]
    vmax = 1
    vmin = 1e-3
    fig, ax = plt.subplots(1,3,figsize=(5, 15), dpi=150)
    for t in range(maxT):
        # TODO: if i do it like this then I need to use the vectorized shit
        # biases are left, down, up, right
        occ = updateOccupancy(occ, t, func)
        print(occ)
        # occ, biases = updateOccupancy(occ, t, func)

        im = ax[t].imshow(occ,cmap=cmap,vmax=vmax,vmin=vmin, aspect='equal')
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
    path = os.path.join(topDir,"modelGraphicNewAdjustedColors.pdf")
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

# don't use this lmao
def modelAndMeasurementGraphic(topDir="/home/fransces/Documents/Figures/Paper"):
    plt.rcParams.update({'font.size': 20, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{bm}'})
    fig, axs = plt.subplot_mosaic("012;333",constrained_layout=True,figsize=(5,5),height_ratios=[1/3,2/3])
    # top 3, model
    cmap = copy.copy(matplotlib.colormaps.get_cmap("viridis"))
    cmap.set_under(color="white")
    cmap.set_bad(color="white")
    maxT = 3
    func = getRandomDistribution('Dirichlet', [1, 1, 1, 1])
    L = 3
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    limits = [L - 3, L + 3]
    vmax = 1
    vmin = 1e-10
    for t in range(maxT):
        # TODO: if i do it like this then I need to use the vectorized shit
        # biases are left, down, up, right
        occ = updateOccupancy(occ, t, func)
        print(occ)
        # occ, biases = updateOccupancy(occ, t, func)
        axs[f"{t}"].imshow(occ,cmap=cmap,vmin=vmin, aspect='equal')
        # ax[t].imshow(occ,cmap=cmap,vmin=vmin,vmax=vmax,aspect='equal')
        [axs[f"{t}"].plot([-1,6],[y+.5,y+.5],'k',lw=1) for y in range(-1,6)]
        [axs[f"{t}"].plot([x+.5,x+.5],[-1,6],'k',lw=1) for x in range(-1,6)]
        axs[f"{t}"].set_xlim(limits)
        axs[f"{t}"].set_ylim(limits)
        axs[f"{t}"].get_xaxis().set_ticks([])
        axs[f"{t}"].get_yaxis().set_ticks([])
        axs[f"{t}"].spines['top'].set_visible(False)
        axs[f"{t}"].spines['right'].set_visible(False)
        axs[f"{t}"].spines['bottom'].set_visible(False)
        axs[f"{t}"].spines['left'].set_visible(False)
    axs["0"].annotate("(a)",xy=(0,1), xycoords='axes fraction', xytext=(+0.5,-0.5),
                 textcoords='offset fontsize', fontsize='small', verticalalignment='top',
                 bbox=dict(facecolor='0.7',edgecolor='none',pad=2.0))
    # bottom fig, measurement
    maxT2 = 500
    L2 = 500
    occ2 = np.zeros((2 * L2 + 1, 2 * L2 + 1))
    occ2[L2, L2] = 1
    for t in range(maxT2):
        occ2 = updateOccupancy(occ2, t, func)
    cmap2 = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap2.set_under(color="white")
    cmap2.set_bad(color="white")
    a = axs["3"].imshow(occ2, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap2, interpolation='none')
    limits2 = [L2 - 100, L2 + 100]
    axs["3"].set_xlim(limits2)
    axs["3"].set_ylim(limits2)
    r = 75
    circle = plt.Circle((L2, L2), r, color='k', ls='--', fill=False, lw=2.5)
    axs["3"].add_patch(circle)
    arrow = FancyArrowPatch((L2, L2), (L2 + r * np.cos(np.pi / 4), L2 + r * np.sin(np.pi / 4)), color='k',
                            mutation_scale=40)
    axs["3"].add_patch(arrow)
    axs["3"].annotate(r"$r(t)$", (500, 545))
    axs["3"].annotate("(b)",xy=(0,1), xycoords='axes fraction', xytext=(+0.5,-0.5),
                 textcoords='offset fontsize', fontsize='small',verticalalignment='top',
                 bbox=dict(facecolor='0.7',edgecolor='none',pad=3.0))
    xvals = np.linspace(400, 600)
    yvals = np.sqrt(r ** 2 - (xvals - L2) ** 2) + L2
    yvals[np.isnan(yvals)] = 500
    y2vals = np.ones(len(xvals)) * 600
    axs["3"].fill_between(xvals, yvals, y2vals, alpha=0.75, hatch='//', facecolor='none', edgecolor='k', linewidth=0)
    axs["3"].fill_between(xvals, 2 * L2 - yvals, np.ones(len(xvals)) * 400, alpha=0.75, hatch='//', facecolor='none',
                    edgecolor='k', linewidth=0)
    fig.colorbar(a,label=r"$\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|\geq r(t)\right)$",location='right')
    axs["3"].get_xaxis().set_ticks([])
    axs["3"].get_yaxis().set_ticks([])
    axs["3"].spines['top'].set_visible(False)
    axs["3"].spines['right'].set_visible(False)
    axs["3"].spines['bottom'].set_visible(False)
    axs["3"].spines['left'].set_visible(False)

    fig.savefig(os.path.join(topDir,"modelAndMeasurement.pdf"),bbox_inches='tight')

def colorsForLambda(lambdaList):
    # normalize values of lambda to be between 0 and 1
    logVals = np.log(lambdaList)
    vals = (logVals - np.min(logVals)) / (np.max(logVals) - np.min(logVals))
    # this goes between teal and green i guess?
    colorList = np.array([[l, 1-l, 1] for l in vals])
    if np.any(np.isnan(colorList)):
        colorList[np.isnan(colorList)] = 0
    return colorList

# TODO: as an inset, plot the mean.
def plotMasterCurve(savePath, statsFileList, tMaxList, lambdaExtVals, markers, verticalLine=True):
    """
    plots given lists of var[lnP] as a function of mastercurve f(lambda,r,t) = lambda r^2/t^2
    """
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    colors = colorsForLambda(lambdaExtVals)
    fig, ax1 = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=150)
    left, bottom, width, height = [0.25, 0.25, 0.4, 0.4]
    # ax2 = fig.add_axes([left, bottom, width, height])
    #plt.figure(figsize=(5, 5), constrained_layout=True, dpi=150)
    ax1.set_xlim([1e-11, 5e2])
    ax1.set_ylim([1e-11, 5e2])
    ax1.set_aspect('equal')
    # ax2.set_aspect('equal')
    # TODO: make more robust? under the assumption that all paths have the same ts list.
    for i in range(len(statsFileList)):
        print(statsFileList[i])
        if verticalLine:
            # this is theoretical prediction
            xLoc = lambdaExtVals[i]
            ax1.loglog([xLoc,xLoc],[xLoc,5e2], color=colors[i],linestyle='solid', zorder=0.0000000001)
        for j in range(len(tMaxList)):
            file = statsFileList[i]
            # label: distribution name
            tempData, label = d.processStats(file)
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= 2))).flatten()
            scalingFuncVals = d.masterCurveValue(tempData[1, :][indices], tempData[2, :][indices],
                                                 tempData[3, :][indices])
            # for main collapse (vlp vs masterfunc linear)
            ax1.loglog(scalingFuncVals, tempData[0, :][indices],
                       markers[i], color=colors[i], markeredgecolor='k',
                       ms=4, mew=0.5, label=label, zorder=np.random.rand())
            # # mean inset
            # with np.errstate(divide='ignore'):
            #     # plot - < lnP > vs t
            #     ax2.loglog(tempData[2,:][indices],-tempData[4,:][indices],markers[i],
            #                color=colors[i],markeredgecolor='k',ms=2,mew=0.25,label=label,
            #                zorder=np.random.rand())
    # for normal mastercurve
    ax1.set_xlabel(r"$\frac{\displaystyle\lambda_{\mathrm{ext}}r^2}{t^2}$")
    ax1.set_ylabel(r"$\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")
    ax1.set_xticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
    ax1.set_yticks([1e-10,1e-8,1e-6,1e-4,1e-2,1e0,1e2])
    # # for the inset
    # ax2.set_xlabel(r"$t$")
    # ax2.set_ylabel(r"$-\mathrm{E}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")

    fig.savefig(f"{savePath}")

def plotMean(savePath, statsFileList, tMaxList, lambdaExtVals, markers):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    colors = colorsForLambda(lambdaExtVals)
    fig, ax = plt.subplots(figsize=(5,5),constrained_layout=True,dpi=150)
    for i in range(len(statsFileList)):
        print(statsFileList[i])
        for j in range(len(tMaxList)):
            file = statsFileList[i]
            # label: distribution name
            tempData, label = d.processStats(file)
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            indices = np.array(np.where((tempData[2, :] == tMaxList[j]) & (tempData[1, :] >= 2))).flatten()

            # mean inset
            with np.errstate(divide='ignore'):
                # x-axis, r^2/ t
                gaussianbehavior = tempData[1, indices]**2 / tempData[2,indices]
                # plot -<lnP> vs r^2/t
                ax.loglog(gaussianbehavior, -tempData[4,:][indices],markers[i],
                           color=colors[i],markeredgecolor='k',ms=4,mew=0.5,label=label,
                           zorder=np.random.rand())
    # prediction
    x = np.logspace(-4,3)
    ax.plot(x,x,color='red')
    # for the inset
    ax.set_xlabel(r"$r(t)^2 / t$")
    ax.set_ylabel(r"$-\mathrm{E}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")
    ax.set_xlim([1e-4, 1e3])
    ax.set_ylim([1e-4, 1e3])
    ax.set_aspect('equal')
    ax.set_xticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])
    ax.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3])
    fig.savefig(f"{savePath}")


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
    savePath = "/home/fransces/Documents/Figures/Paper/2DRWREMasterCurveWithInset.pdf"

    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])

    # for dropping distributions which have very close values of lambda
    minSavePath = "/home/fransces/Documents/Figures/Paper/2DRWREMasterCurveReduced.pdf"
    minimalStatsList = [path31, pathCorner, path1, pathDelta, path03, path01, path003]
    minExpVarXList, minLambdaList = d.getListOfLambdas(minimalStatsList)
    minmarkers = ['o'] + ['s'] + ['o'] + ['v'] + ['o']*3
    plotMasterCurve(minSavePath, minimalStatsList, tMaxList, markers=minmarkers, lambdaExtVals=minLambdaList, verticalLine=True)

    meanPath = "/home/fransces/Douments/Figures/Paper/2DRWREMean.pdf"
    plotMean(meanPath, minimalStatsList, tMaxList, markers=minmarkers, lambdaExtVals=minLambdaList)