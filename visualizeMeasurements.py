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


def generateGifRWRE(occupancy, maxT, alphas, pathName,startT=1, listOfTimes=None):
    os.makedirs(pathName,exist_ok=True)
    func = m.getRandomDistribution("Dirichlet", alphas)
    # L = occupancy.shape[0]//2
    # if listOfTimes is None:
    #     listOfTimes = np.unique(np.geomspace(startT, maxT, num=10).astype(int))
    for t in range(startT, maxT):
        occupancy = m.updateOccupancy(occupancy, t, func)
        # if t in listOfTimes:
        a = np.copy(occupancy)
        if t%2 == 0:
            if t < 10:
                tStr = f"00{t}"
            elif 10 <= t < 100:
                tStr = f"0{t}"
            else:
                tStr = f"{t}"
            plt.imshow(a)
            plt.savefig(pathName+tStr+'.png',bbox_inches='tight')

def generateGifSSRW(occupancy, maxT, pathName):
    """
    example:
        L = 100; tMaX = 500; alphas = np.array([0.5]*4);occ=np.zeros((2*L+1,2*L+1));occ[L,L] = 1
        v.generateGifSSRW(occ, tMaX)
    """
    os.makedirs(pathName,exist_ok=True)
    func = m.getRandomDistribution("OneQuarter","")
    # L = occupancy.shape[0]//2
    for t in range(1, maxT):
        occupancy = m.updateOccupancy(occupancy, t, func)
        a = np.copy(occupancy)
        if t%2 == 0:
            if t < 10:
                tStr = f"00{t}"
            elif 10 <= t < 100:
                tStr = f"0{t}"
            else:
                tStr = f"{t}"
            plt.imshow(a)
            plt.savefig(pathName+tStr+".png",bbox_inches='tight')

# can't use this anymore because some files are h5 and some are npy
def collapseAlphaAndVar(paths, alphaList, scaling=3):
    """
    assumes stats.npz files are already created

    paths: list of strs defining where to get data
    alphaList: list of floats defining the alphas; order corresponds to paths
    scaling: int (0-3) corresponding to scaling vt, vt^1/2, vt/ln(t), vt/sqrt(ln(t)
        defaults to vt/sqrt(ln(t))

    returns plot with expected scaling of var(ln(prob) / (v^2 / (2pi^3*alpha))
    """
    plt.ion()
    plt.figure(figsize=(5, 4), constrained_layout=True, dpi=150)
    plt.yscale("log")
    plt.xscale("log")
    # assumes /data/memwhatever/dirichlet/ALPHAsomething/L5000/tMax10000
    for i in len(paths):
        info = np.load(f"{paths[i]}/info.npz")  # times, velocities..
        times = info['times']
        velocities = info['velocities'].flatten()
        path = paths[i]
        temp = np.load(f"{path}/stats.npz")
        # get the tMax var for scaling we're interested in, for every velocity
        lastVars = temp['variance'][scaling,-1,:]
        # TODO: easy way to get alpha info?
        # label =
        plt.plot(velocities,lastVars/(velocities**2/(2*alphaList[i]*np.pi**3)),
                 label=f"alpha = {alphaList[i]}")
        # means.append(temp['mean'])
        # vars.append(temp['variance'])
    plt.xlabel("velocities")
    plt.ylabel("r$\frac{2\pi^3\alpha\cdot var(\log(p))}{v^2}$")
    plt.title(f"prob past sphere moving at scaling idx {scaling} for L=5000 tMax=10000"
              f"\n dirichlet distribution")


def visualizeAlphaAndVar(savePath,regime='tOnSqrtLogT'):
    # scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
    if regime =='tOnSqrtLogT':  # default to crit regime
        regimeIdx = 3
    elif regime =='sqrt':
        regimeIdx = 1
    elif regime =='linear':
        regimeIdx = 0
    os.makedirs(savePath,exist_ok=True)
    path003 = "data/memoryEfficientMeasurements/dirichlet/ALPHA0.03162278/L5000/tMax10000"
    path01 = "data/memoryEfficientMeasurements/dirichlet/ALPHA0.1/L5000/tMax10000"
    path03 = "data/memoryEfficientMeasurements/dirichlet/ALPHA0.31622777/L5000/tMax10000"
    path1 = "data/memoryEfficientMeasurements/dirichlet/ALPHA1/L5000/tMax10000"
    path3 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000/"
    path10 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000"
    path31 = "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000/"
    var003 = np.load(f"{path003}/stats.npz")['variance']
    var01 = np.load(f"{path01}/stats.npz")['variance']
    var03 = np.load(f"{path03}/stats.npz")['variance']
    var1 = np.load(f"{path1}/stats.npz")['variance']
    var3 = h5py.File(f"{path3}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array
    var10 = h5py.File(f"{path10}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array
    var31 = h5py.File(f"{path31}/Stats.h5", "r")[regime]['var'][:, :]  # NP Array

    # grab list of velocities for small alpha (1e-3 to 10)
    info = np.load(f"{path03}/info.npz")
    time = info['times']
    velsSmallAlpha = info['velocities'].flatten()
    # grab list of velocities for big alpha (1e-5 to 10)
    with open(f"{path10}/variables.json","r") as v:
        variables = json.load(v)
    velsBigAlpha = np.array(variables['velocities'])
    # indexing to get time
    idx = [-2, -25, -50, -75, -100, -125, -150, -175, -200, -225, -250, -275, -300]
    x = np.logspace(-10, 0)
    plt.ion()
    n = 7
    colors = plt.cm.jet(np.linspace(0, 1, n))
    alphas = [0.03162278, 0.1, 0.31622777, 1, 3.1622776, 10, 31.622776]
    VarEX = []  # 0.03, 0.1, 0.3, 1, 3, 10, 31
    for i in range(n):
        print(f"alpha: {alphas[i]}")
        params = np.array([alphas[i]] * 4)
        VarEX.append(m.getExpVarX('Dirichlet', params))
        print(f"VarEX: {VarEX}")
    for timeIDX in idx:
        t = time[timeIDX]
        print(f"time: {t}")
        if t < 100:
            tStr = f"00{t}"
        elif 100 <= t < 1000:
            tStr = f"0{t}"
        else:
            tStr = f"{t}"
        lastVar003 = var003[regimeIdx, timeIDX, :]
        lastVar01 = var01[regimeIdx, timeIDX, :]
        lastVar03 = var03[regimeIdx, timeIDX, :]
        lastVar1 = var1[regimeIdx, timeIDX, :]
        lastVar3 = var3[timeIDX, :]
        lastVar10 = var10[timeIDX, :]
        lastVar31 = var31[timeIDX, :]
        if regime == 'tOnSqrtLogT':  # default to crit regime
            prefactor = (1 / (4 * np.pi))
            xLabel = r"$\frac{1}{4\pi}v^2 \frac{Var_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-Var_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
            title = r"$r(t)\propto\frac{vt}{\sqrt{\log{t}}}"
        elif regime == 'sqrt':
            prefactor = (np.log(t) / (4 * np.pi * t))
            xLabel = r"$\frac{\log{t}}{4\pi t}v^2 \frac{Var_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-Var_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
            title = r"$r(t)\propto vt^{1/2}$"
        elif regime == 'linear':
            prefactor = (np.log(t) / (4 * np.pi))
            xLabel = r"$\frac{\log{t}}{4\pi}v^2 \frac{Var_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}{1-Var_{\nu}(\mathbb{E}^{\xi}[\vec{X}])}$"
            title = r"$r(t)\propto vt$"
        plt.figure(figsize=(5, 4), constrained_layout=True, dpi=150)
        plt.loglog(prefactor * (VarEX[0] / (1 - VarEX[0])) * velsSmallAlpha ** 2, lastVar003, '.', color=colors[0], label=r"$\alpha= 0.03$")
        plt.loglog(prefactor * (VarEX[1] / (1 - VarEX[1])) * velsSmallAlpha ** 2, lastVar01, '.', color=colors[1], label=r"$\alpha= 0.1$")
        plt.loglog(prefactor * (VarEX[2] / (1 - VarEX[2])) * velsSmallAlpha ** 2, lastVar03, '.', color=colors[2], label=r"$\alpha= 0.3$")
        plt.loglog(prefactor * (VarEX[3] / (1 - VarEX[3])) * velsSmallAlpha ** 2, lastVar1, '.', color=colors[3], label=r"$\alpha= 1$")
        plt.loglog(prefactor * (VarEX[4] / (1 - VarEX[4])) * velsBigAlpha ** 2, lastVar3, '.', color=colors[4], label=r"$\alpha= 3$")
        plt.loglog(prefactor * (VarEX[5] / (1 - VarEX[5])) * velsBigAlpha ** 2, lastVar10, '.', color=colors[5], label=r"$\alpha= 10$")
        plt.loglog(prefactor * (VarEX[6] / (1 - VarEX[6])) * velsBigAlpha ** 2, lastVar31, '.', color=colors[6], label=r"$\alpha= 31$")
        plt.plot(x, x, color='k', linestyle='dashed', label=r"y=x")
        plt.ylim([10 ** -8, 10 ** 3])
        plt.xlim([10 ** -8, 10 ** 3])
        plt.xlabel(xLabel)
        plt.ylabel(fr"Var($\log P $) at t={t}")
        plt.title(title)
        plt.legend(loc=2)
        plt.savefig(f"{savePath}/" + tStr + ".png")


def createMeasurementGraphic(saveFile):
    """
    savefile: str, path to which fig is saved
    """
    # os.makedirs(saveFile,exist_ok=True)
    # plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

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

    arrow1 = matplotlib.patches.FancyArrowPatch((maxT, maxT), (maxT + r1 * np.cos(-np.pi / 3), maxT + r1 * np.sin(-np.pi / 3)), mutation_scale=mutation_scale, color='k')
    arrow2 = matplotlib.patches.FancyArrowPatch((maxT, maxT), (maxT + r2 * np.cos(np.pi / 6), maxT + r2 * np.sin(np.pi / 6)), mutation_scale=mutation_scale, color='k')
    arrow3 = matplotlib.patches.FancyArrowPatch((maxT, maxT), (maxT, maxT + r3), mutation_scale=mutation_scale, color='k')

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

    # setting norm=matplotlib.colors.LogNorm automatically takes the log (base 10)
    # trying to do like np.log(occ) and not do the norm gets back the terrible checkerboard
    ax.imshow(occ, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap,  interpolation='gaussian', alpha=0.75)
    fig.savefig(f"{saveFile}.pdf", bbox_inches='tight')


def plotAllSystems(path,saveDir, regime='tOnSqrtLogT',takeLog=False):
    files = os.listdir(path)
    os.makedirs(saveDir,exist_ok=True)
    files.remove('info.npz')
    files.remove('statsNoLog.npz')
    files.remove('stats.npz')
    info = np.load(f"{path}/info.npz")
    velocities = info['velocities'].flatten()
    vIdx = -8  # hard code in 0.39 for v, for figure purposes
    time = info['times']

    if regime =='tOnSqrtLogT':  # default to crit regime
        regimeIdx = 3
        r = velocities[vIdx] * time / (np.sqrt(np.log(time)))
    elif regime =='sqrt':
        regimeIdx = 1
        r = velocities[vIdx] * (time ** (1/2))
    elif regime =='linear':
        regimeIdx = 0
        r = velocities[vIdx] * time
    if takeLog:
        statsMean = np.load(f"{path}/stats.npz")['mean'][regimeIdx, :, vIdx]
        ylabel = r"$-\log{P(r(t))}$"
        meanLabel = r"-$\langle \log{P(r(t))} \rangle$"
    else:
        statsMean = np.load(f"{path}/statsNoLog.npz")['mean'][regimeIdx, :, vIdx]
        ylabel = r"$P(r(t))$"
        meanLabel = r"$\langle P(r(t))\rangle$"
    fig, ax = plt.subplots(figsize=(5,4),dpi=150,constrained_layout=True)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    for file in files:
        temp = np.load(f"{path}/{file}")[regimeIdx,:,vIdx]
        #TODO: what if dots plus lines instead
        if takeLog:
            with np.errstate(divide='ignore'):
                ax.loglog(time,-np.log(temp),'.',linewidth=1,alpha=0.4)
        else:
            ax.semilogx(time, temp,'.',linewidth=1,alpha=0.4)
    if takeLog:
        with np.errstate(divide='ignore'):
            ax.loglog(time,- statsMean,'o',color='k',alpha=0.8, label=meanLabel)
            prediction = r**2 / time
            ax.loglog(time, prediction, color='red',linestyle='dashed',label=r"$\frac{r(t)^2}{t}$")
        savePath = f"{saveDir}/{regime}"+"AllSystemsLnP.png"
    else:
        ax.semilogx(time,statsMean,'o',color='k',label=meanLabel)
        savePath = f"{saveDir}/{regime}"+"AllSystemsRawP.png"
    ax.legend()
    #ax.set_ylim([1e-6,1])
    # TODO: save as png because too many points
    fig.savefig(savePath,bbox_inches='tight')

# all old shit because we've changed stuff.
# def getLastValsLog(path,velocityIdx,savePath):
#     """
#     path: str; specify path to data
#     velocity; int; needs to be index of the velocity at which you want to probe
#     scaling; int; 0 = linear, 1 = sqrt, 2= vt/ln(t); 3 = vt/sqrt(ln(t)); 4 = fixed radius
#     savePath: str; specify path to which pngs saved
#     """
#     info = np.load(f"{path}/info.npz")
#     times = info['times']
#     velocities = info['velocities'].flatten()
#     os.makedirs(os.path.join(savePath,"distribution",f"v{velocities[velocityIdx]}"),exist_ok=True)
#     os.makedirs(os.path.join(savePath,"cumulative"),exist_ok=True)
#     saveFile = os.path.join(savePath,"distribution",f"v{velocities[velocityIdx]}","lastValsLog")
#     # cumulPath = os.path.join(savePath,"cumulative")
#     x = np.linspace(-4,4,50000)
#     p = norm.pdf(x,0,1)
#     c = norm.cdf(x,0,1)
#     # only get data files
#     fileList = glob.glob("*.npy",root_dir=path)
#     # print(f"time for glob: {time.time()-s}")
#     lastVT = []
#     lastVSqrtT = []
#     lastVTOnSqrtLogT = []
#     # s = time.time()
#     for file in fileList:
#         data = np.load(f"{path}/{file}")
#         # TODO: implement thing so it doesn't just grab
#         #  the last val. as a proxy for maxT...
#         valsAtVelocityVT = data[0,:,velocityIdx]
#         valsAtVelocityVSqrtT = data[1,:,velocityIdx]
#         valsAtVelocityVTOnSqrtLogT = data[3,:,velocityIdx]
#         lastVT.append(valsAtVelocityVT[-1])
#         lastVSqrtT.append(valsAtVelocityVSqrtT[-1])
#         lastVTOnSqrtLogT.append([-1])
#         # lastVT.append(valsAtVelocityVT[np.max(np.where(valsAtVelocityVT != 0))])
#         # lastVSqrtT.append(valsAtVelocityVSqrtT[np.max(np.where(valsAtVelocityVSqrtT != 0))])
#         # lastVTOnSqrtLogT.append([np.max(np.where(valsAtVelocityVTOnSqrtLogT != 0))])
#         # logProbVT.append(np.log(valsAtVelocityVT[np.max(np.where(~np.isnan(valsAtVelocityVT)))]))
#         # logProbVSqrtT.append(np.log(valsAtVelocityVSqrtT[np.max(np.where(~np.isnan(valsAtVelocityVSqrtT)))]))
#         # logProbCrit.append(np.log(valsAtVelocityCrit[np.max(np.where(~np.isnan(valsAtVelocityCrit)))]))
#     np.savez_compressed(saveFile,VT=lastVT,VSqrtT=lastVSqrtT,VTOnSqrtLogT=lastVTOnSqrtLogT)
#
# def getHistogram(histogramPath):
#     lastLogVals = np.load(histogramPath)
#     VT = lastLogVals['VT']
#     VSqrtT = lastLogVals['VSqrtT']
#     VTOnSqrtLogT = lastLogVals['VTOnSqrtLogT']
#     x = np.linspace(-3,3,50000)
#     p = norm.pdf(x,0,1)
#     c = norm.cdf(x,0,1)
#     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5,4),dpi=150)
#     ax1.set_yscale("log")
#     # # print(f"time to do entire for loop: {time.time() - s}")
#     # # s = time.time()
#     # mean, var = np.mean(logProb), np.var(logProb)
#     # #print(f"time it takes to do mean var calc: {time.time()-s}")
#     # plt.figure(figsize=(5,4),constrained_layout=True,dpi=150)
#     # plt.yscale("log")
#     # plt.hist((logProb-mean)/np.sqrt(var),bins=500,density=True)
#     # plt.semilogy(x,p,color='black')
#     # # plt.title(f"centered and scaled histogram of ln(prob) "
#     # #          f"\n at t={t} with v={velocity} \n from {path} "
#     # #          f"\n with gaussian in black")
#     # plt.savefig(saveNameDistribution)
#     # plt.figure(figsize=(5,4),constrained_layout=True,dpi=150)
#     # plt.plot(np.sort((logProb-mean)/np.sqrt(var)),np.arange(len(logProb))/len(logProb),label="RWRE")
#     # plt.plot(x,c,color='black',label="Normal")
#     # #plt.title(f"cdf of ln(prob) at t={t} with velocity={velocity}"
#     # #          f"\n from {path}"
#     # #          f"\n with gaussian in black")
#     # plt.savefig(saveNameCumul)
#
#
#
# # TODO: fix because switched format
# # to get histograms of ln(probs)
# def visualizeDistributionOld(path,velocity,t,savePath):
#     if not os.path.isdir(savePath):
#         os.makedirs(savePath)
#     x = np.linspace(-4,4,50000)
#     p = norm.pdf(x,0,1)
#     c = norm.cdf(x,0,1)
#     s = time.time()
#     filelist = glob.glob("sphere*",root_dir=path)
#     print(f"time for glob: {time.time()-s}")
#     logProb = []
#     s = time.time()
#     for file in filelist:
#         data = pd.read_csv(f"{path}/{file}")
#         # TODO: implement thing so it doesn't just grab
#         #  the last val. as a proxy for maxT...
#         val = np.array(data[velocity])[-1]
#         logProb.append(np.log(val))
#     print(f"time to do entire for loop: {time.time() - s}")
#     s = time.time()
#     mean, var = np.mean(logProb), np.var(logProb)
#     print(f"time it takes to do mean var calc: {time.time()-s}")
#     plt.figure(figsize=[9.6,7.2])
#     plt.yscale("log")
#     plt.hist((logProb-mean)/np.sqrt(var),bins=500,density=True)
#     plt.semilogy(x,p,color='black')
#     plt.title(f"centered and scaled histogram of ln(prob) "
#               f"\n at t={t} with v={velocity} \n from {path} "
#               f"\n with gaussian in black")
#     plt.savefig(os.path.join(savePath,"vel"+velocity,"distribution.png"),dpi=150)
#     plt.figure(figsize=[9.6,7.2])
#     plt.plot(np.sort((logProb-mean)/np.sqrt(var)),np.arange(len(logProb))/len(logProb))
#     plt.plot(x,c,color='black')
#     plt.title(f"cdf of ln(prob) at t={t} with velocity={velocity}"
#               f"\n from {path}"
#               f"\n with gaussian in black")
#     plt.savefig(os.path.join(savePath,"vel"+velocity,"cdf.png"),dip=150)
#     np.save(os.path.join(savePath,"lastProbs.txt"),logProb)
#
# # figs for comps?
# # plt.figure(figsize=(5,4),constrained_layout=True,dpi=150)
# # n =
# # colors = plt.cm.jeet(np.linspace(0,1,n))
# # for i in range(n):
# #       plt.loglog(time[2<(vels[i]*time/(np.sqrt(np.log(time))))], varLog[3,:,i][2<(vels[i]*time/(np.sqrt(np.log(time))))], color=colors[i])
# # plt.legend(vels)
# # plt.xlabel("Time");plt.ylabel(r"Var($\log{P}$)")
# # plt.savefig("/home/fransces/Documents/code/extremeDiffusionND/compFigs/titlehere.png")