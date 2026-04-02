import numpy as np
from matplotlib import pyplot as plt
import os
import json
from memEfficientEvolve2DLattice import getExpVarXDotProduct
from scipy.special import erfc

# for past a line and past a point
def plotAllStatsPastLinePastPoint():
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    # load in stats
    path = "/mnt/talapasData/data/pastLineTake3/1000/Line/"
    savePath = "/home/fransces/Documents/Figures/"
    with open(f"{path}variables.json", "r") as v:
        variables = json.load(v)
    tMax = variables['tMax']
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    n = 50
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    # past line
    print("past line!")
    statsLine = np.load(os.path.join(path, "LineStats.npy"))  # mean, var, skew, kurtosis
    # mean
    lineFig0, lineAx0 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        lineAx0[0].loglog(times, - statsLine[0, :, i], '.', ms=1, color=colors[i])
    lineAx0[0].set_xlabel(r"$t$")
    lineAx0[0].set_ylabel(r"$-\langle \ln{(P_{line})} \rangle$")
    lineAx0[0].set_title(r"past line $vt^{1/2}$, v=1e2 (blue) to 3 (yellow)")
    for i in range(50, 100):  # t/sqrt(ln(t))
        lineAx0[1].loglog(times, - statsLine[0, :, i], '.', ms=1, color=colors[i - 50])
    lineAx0[1].set_xlabel(r"$t$")
    lineAx0[1].set_ylabel(r"$-\langle \ln{(P_{line})} \rangle$")
    lineAx0[1].set_title(r"past line $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(100, 150):  # t
        lineAx0[2].loglog(times, - statsLine[0, :, i], '.', ms=1, color=colors[i - 100])
    lineAx0[2].set_xlabel(r"$t$")
    lineAx0[2].set_ylabel(r"$- \langle \ln{(P_{line})}\rangle$")
    lineAx0[2].set_title(r"past line $vt$, v=1e-2 (blue) to 3 (yellow)")
    lineFig0.savefig(os.path.join(savePath, "lineMean30k.png"))

    # var
    lineFig1, lineAx1 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        lineAx1[0].loglog(times, statsLine[1, :, i], '.', ms=1, color=colors[i])
    lineAx1[0].set_xlabel(r"$t$")
    lineAx1[0].set_ylabel(r"$\mathrm{Var}\left[\ln{(P_{line})}\right]$")
    lineAx1[0].set_title(r"past line $vt^{1/2}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(50, 100):  # t/sqrt(ln(t))
        lineAx1[1].loglog(times, statsLine[1, :, i], '.', ms=1, color=colors[i - 50])
    lineAx1[1].set_xlabel(r"$t$")
    lineAx1[1].set_ylabel(r"$\mathrm{Var}\left[\ln{(P_{line})}\right]$")
    lineAx1[1].set_title(r"past line  $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(100, 150):  # t
        lineAx1[2].loglog(times, statsLine[1, :, i], '.', ms=1, color=colors[i - 100])
    lineAx1[2].set_xlabel(r"$t$")
    lineAx1[2].set_ylabel(r"$\mathrm{Var}\left[\ln{(P_{line})}\right]$")
    lineAx1[2].set_title(r"past line $vt$, v=1e-2 (blue) to 3 (yellow)")
    lineFig1.savefig(os.path.join(savePath, "lineVar30k.png"))

    # skew
    lineFig2, lineAx2 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        lineAx2[0].loglog(times, - statsLine[2, :, i], '.', ms=1, color=colors[i])
    lineAx2[0].set_xlabel(r"$t$")
    lineAx2[0].set_ylabel(r"$-\mathrm{Skew}\left[\ln{(P_{line})}\right]$")
    lineAx2[0].set_title(r"past line $vt^{1/2}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(50, 100):  # t/sqrt(ln(t))
        lineAx2[1].loglog(times, -statsLine[2, :, i], '.', ms=1, color=colors[i - 50])
    lineAx2[1].set_xlabel(r"$t$")
    lineAx2[1].set_ylabel(r"$-\mathrm{Skew}\left[\ln{(P_{line})}\right]$")
    lineAx2[1].set_title(r"past line  $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(100, 150):  # t
        lineAx2[2].loglog(times, -statsLine[2, :, i], '.', ms=1, color=colors[i - 100])
    lineAx2[2].set_xlabel(r"$t$")
    lineAx2[2].set_ylabel(r"$-\mathrm{Skew}\left[\ln{(P_{line})}\right]$")
    lineAx2[2].set_title(r"past line $vt$, v=1e-2 (blue) to 3 (yellow)")
    lineFig2.savefig(os.path.join(savePath, "lineSkew30k.png"))

    # past point
    print("past point!)")
    statsPoint = np.load(os.path.join(path, "PointStats.npy"))  # mean, var, skew, kurtosis
    # mean
    pointFig0, pointAx0 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        pointAx0[0].loglog(times, - statsPoint[0, :, i], '.', ms=1, color=colors[i])
    pointAx0[0].set_xlabel(r"$t$")
    pointAx0[0].set_ylabel(r"$-\langle \ln{(P_{line})} \rangle$")
    pointAx0[0].set_title(r"past point $vt^{1/2}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(50, 100):  # t/sqrt(ln(t))
        pointAx0[1].loglog(times, - statsPoint[0, :, i], '.', ms=1, color=colors[i - 50])
    pointAx0[1].set_xlabel(r"$t$")
    pointAx0[1].set_ylabel(r"$-\langle \ln{(P_{line})} \rangle$")
    pointAx0[1].set_title(r"past point $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(100, 150):  # t
        pointAx0[2].loglog(times, - statsPoint[0, :, i], '.', ms=1, color=colors[i - 100])
    pointAx0[2].set_xlabel(r"$t$")
    pointAx0[2].set_ylabel(r"$- \langle \ln{(P_{line})} \rangle $")
    pointAx0[2].set_title(r"past point $vt$, v=1e-2 (blue) to 3 (yellow)")
    pointFig0.savefig(os.path.join(savePath, "pointMean30k.png"))
    # var
    pointFig1, pointAx1 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        pointAx1[0].loglog(times, statsPoint[1, :, i], '.', ms=1, color=colors[i])
    pointAx1[0].set_xlabel(r"$t$")
    pointAx1[0].set_ylabel(r"$\mathrm{Var}\left[\ln{(P_{line})}\right]$")
    pointAx1[0].set_title(r"past point $vt^{1/2}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(50, 100):  # t/sqrt(ln(t))
        pointAx1[1].loglog(times, statsPoint[1, :, i], '.', ms=1, color=colors[i - 50])
    pointAx1[1].set_xlabel(r"$t$")
    pointAx1[1].set_ylabel(r"$\mathrm{Var}\left[\ln{(P_{line})}\right]$")
    pointAx1[1].set_title(r"past point  $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(100, 150):  # t
        pointAx1[2].loglog(times, statsPoint[1, :, i], '.', ms=1, color=colors[i - 100])
    pointAx1[2].set_xlabel(r"$t$")
    pointAx1[2].set_ylabel(r"$\mathrm{Var}\left[\ln{(P_{line})}\right]$")
    pointAx1[2].set_title(r"past point $vt$, v=1e-2 (blue) to 3 (yellow)")
    pointFig1.savefig(os.path.join(savePath, "pointVar30k.png"))
    # skew
    pointFig2, pointAx2 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        pointAx2[0].loglog(times, -statsPoint[2, :, i], '.', ms=1, color=colors[i])
    pointAx2[0].set_xlabel(r"$t$")
    pointAx2[0].set_ylabel(r"$-\mathrm{Skew}\left[\ln{(P_{line})}\right]$")
    pointAx2[0].set_title(r"past point $vt^{1/2}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(50, 100):  # t/sqrt(ln(t))
        pointAx2[1].loglog(times, -statsPoint[2, :, i], '.', ms=1, color=colors[i - 50])
    pointAx2[1].set_xlabel(r"$t$")
    pointAx2[1].set_ylabel(r"$-\mathrm{Skew}\left[\ln{(P_{line})}\right]$")
    pointAx2[1].set_title(r"past point $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 3 (yellow)")
    for i in range(100, 150):  # t
        pointAx2[2].loglog(times, - statsPoint[2, :, i], '.', ms=1, color=colors[i - 100])
    pointAx2[2].set_xlabel(r"$t$")
    pointAx2[2].set_ylabel(r"$- \mathrm{Skew}\left[\ln{(P_{line})}\right]$")
    pointAx2[2].set_title(r"past point $vt$, v=1e-2 (blue) to 3 (yellow)")
    pointFig2.savefig(os.path.join(savePath, "pointSkew30k.png"))
    return


#for line and point
def processLinePointStatsNPY(fileName):
    """ take npy file with the saved, calculated stats of systems and outputs an
    array of [mean[lnP], var[lnP], skew[lnpP], kurtosis[lnP], r^2, time].
    note that this takes every r,t, pair and disregards any scaling.
    basically it's the same data but we're putting it into a different format
    specific to the dirichlet alpha1 """
    stats = np.load(fileName)
    topDir = os.path.split(fileName)[0]  # /mnt/talapasData/data/pastLine/1000/Line/
    with open(f"{topDir}/variables.json", "r") as v:
        variables = json.load(v)
    times = np.unique(np.geomspace(1, variables['tMax'], 500).astype(int))
    velocities = np.array(variables['velocities'])
    # print(velocities)
    # with open(f"/home/fransces/Documents/code/extremeDiffusionND/histogramPaper/h5pyVariables.json", "r") as v:
    #     variables = json.load(v)
    # times = np.array(variables['ts'])
    # print(times)
    # this one is 1d aray by 39300 (262 * 150)
    longTs = np.tile(times, 3 * velocities.shape[0])  # 1,2,...,1000,1,2,...,1000,...
    with np.errstate(invalid='ignore', divide='ignore'):
        sqrtRadii = ((velocities * np.expand_dims(np.sqrt(times), 1)))
        criticalRadii = ((velocities * np.expand_dims(times / np.sqrt(np.log(times)), 1)))
        linearRadii = ((velocities * np.expand_dims(times, 1)))
    #  262 by 150 flattened to 1d array of length 39300
    radii = (np.hstack((sqrtRadii, criticalRadii, linearRadii))).flatten("F")

    # this was for the h5 data
    # radii = np.hstack((linearRadii, criticalRadii, sqrtRadii)).flatten("F")
    # data = np.array([stats[0, :, :].flatten("F"), stats[1, :, :].flatten("F"),
    #                  stats[2, :, :].flatten("F"), stats[3, :, :].flatten("F"),
    #                  radiiSq, longTs])

    # ok so the issue is how i did this??? maybe???
    # but i swear i tested it more. and i think the flatten gives the same thing...
    data = np.array([stats[0, :, :].flatten("F"), stats[1, :, :].flatten("F"),
                     stats[2, :, :].flatten("F"), radii, longTs])
    return data  # 6 by 39300 array


def plotDifferentScalings(topDir, savePath, lambdaExt, rMin=3):
    print('comparing scalings')
    plt.ion()
    # i need to recreate that one fig. with the blue /green / purple
    # plots of scaled var[ln[p]] vs t with different scalings
    # it doesn't actually matter which scalings we use, just the pairs of rs and ts
    # path = "/mnt/talapasData/data/pastLine/1000/Line/"
    # statsFileList = [os.path.join(path, "LineStats.npy"), os.path.join(path, "PointStats.npy")]
    #statsFile = os.path.join(path, "LineStats.npy")
    statsFile = os.path.join(topDir, "LineStats.npy")
    with open(f"{topDir}variables.json","r") as v:
        variables = json.load(v)
    tMax = variables['tMax']
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    # to re-create h5py past circle results
    # path = "/home/fransces/Documents/code/extremeDiffusionND/histogramPaper/"
    # statsFileList = ["h5StatsAsNpy.npy"]
    # with open(f"{path}h5pyVariables.json", "r") as v:
    #     variables = json.load(v)
    # times = np.array(variables['ts'])
    plt.figure()
    # load in all data
    fileName = statsFile
    plt.title(f"{os.path.split(fileName)[1]}, \n rMin={rMin}")
    data = processStatsNPY(fileName)  # mean, var, skew, r, t
    # mean = data[0,:]
    var = data[1, :]
    # skew = data[2,:]
        # kurtosis = data[3,:]
    rs = data[3,:]  # for the circle h5 data
    ts = data[4,:]  # for the circlee h5 data
    # ts = data[5, :]
    indices = (rs > rMin)
    # indices = (rs**2 / ts > 10) & (rs > rMin)
    # recalc velocities
    vLin = rs / ts
    vMod = rs * np.sqrt(np.log(ts)) / ts
    vSqrt = rs / np.sqrt(ts)

    # predictions without the logscale calculation
    # note that these need to be compared to plotting (1/v^2) var[lnP] w/ time
    linPred = lambdaExt * np.ones_like(times)
    modPred = lambdaExt * (1 / np.log(times))
    sqrtPred = lambdaExt * (1 / times)

    # # predictions wtih the logscale calculation
    # linPred = lambdaExt * np.log(times)
    # modPred = lambdaExt * np.ones_like(times)
    # sqrtPred = lambdaExt * np.log(times) / times

    # plot data
    plt.loglog(ts[indices], var[indices] / (lambdaExt * vLin[indices] ** 2),linewidth=0.5,color='darkgreen', alpha=0.5)
    plt.loglog(ts[indices], var[indices] / (lambdaExt * vMod[indices] ** 2),linewidth=0.5, ms=1,color='dodgerblue', alpha=0.5)
    plt.loglog(ts[indices], var[indices] / (lambdaExt * vSqrt[indices] ** 2), linewidth=0.5,ms=1,color='darkslateblue', alpha=0.5)
    # plot predicitons w/o logscale
    plt.loglog(times, linPred, color='darkgreen', linestyle='dotted', label=r" (linear prediction) $\lambda_{\mathrm{ext}}$")
    plt.loglog(times, modPred, color='dodgerblue', linestyle='dashed', label=r"(moderate deviation prediction) $\lambda_{\mathrm{ext}}/\ln{t}$")
    plt.loglog(times, sqrtPred, color='darkslateblue', linestyle='dashdot', label=r"(sqrt prediction) $\lambda_{\mathrm{ext}}/t$")
    # # plot with logscale
    # plt.loglog(times, linPred, color='darkgreen', linestyle='dotted', label=r" (linear prediction) $c\lambda_{\mathrm{ext}} \ln{t}$")
    # plt.loglog(times, modPred, color='dodgerblue', linestyle='dashed', label=r"(moderate deviation prediction) $c\lambda_{\mathrm{ext}}$")
    # plt.loglog(times, sqrtPred, color='darkslateblue', linestyle='dashdot', label=r"(sqrt prediction) $c\lambda_{\mathrm{ext}} \ln{t}/t$")

    plt.title(rf"$r>{rMin}$  {statsFile}")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\frac{\mathrm{Var}[\ln{P]}}{v_{scaling}^2}$")
    plt.legend(fontsize='x-small')
    # plt.savefig(f"{savePath}/cirlceH5pyScalings.png")
    # plt.savefig(f"{savePath}/scalingComparisons{i}WithCutoff.png")
    plt.savefig(f"{savePath}/scalingComparisons30kSystems.png")
    return


def collapseMean(topDir, savePath, rMin=3):
    print('mean collapse')
    # path = "/mnt/talapasData/data/pastLine/1000/Line/"
    fig, ax = plt.subplots(constrained_layout=True)

    # load in the stats for the 30k systems
    statsFile = os.path.join(topDir, "LineStats.npy")
    data = processStatsNPY(statsFile)  # mean, var, sskew, r, t
    mean = data[0,:]
    r = data[3,:]
    t = data[4, :]
    indices = (r > rMin)
    # indices = (r**2 / t > 10) & (r > rMin)
    x = np.logspace(0,3)
    with np.errstate(divide='ignore'):
        # gaussianBehavior = r**2 / t  # -mean vs r^2/t should be a straight line
        meanLineBehavior = np.log(erfc((r-1)/(np.sqrt(t-1)))/2)
        ax.loglog(-meanLineBehavior[indices], -mean[indices], linewidth=0.5, alpha=0.7)
        # ax.loglog(gaussianBehavior[indices], -mean[indices],linewidth=0.5,alpha=0.7)
    # ax.set_xlabel(r"$r^2/t$")
    ax.set_xlabel(r"$-\ln{(\frac{1}{2}\mathrm{erfc}((r-1)/\sqrt{t-1}))}$")  # for past a line
    ax.set_ylabel(r"$-\mathbb{E}_{\nu}[\ln{P}]$")
    ax.set_title(f"all data past line,regardless of scaling \n r>{rMin} \n {statsFile}")

    # SSRW
    ssrwFileName = "/home/fransces/Documents/code/extremeDiffusionND/tests/dynamicRange/SSRW/projects/L1000/Line/Final0.npy"
    SSRW = np.load(ssrwFileName)  # 262 by 150
    with open("/home/fransces/Documents/code/extremeDiffusionND/tests/dynamicRange/SSRW/projects/L1000/Line/variables.json", "r") as v:
        variables = json.load(v)
    times = np.unique(np.geomspace(1, variables['tMax'], 500).astype(int))
    velocities = np.array(variables['velocities'])
    longTs = np.tile(times, 3 * velocities.shape[0])  # 1,2,...,1000,1,2,...,1000,...
    with np.errstate(invalid='ignore', divide='ignore'):
        sqrtRadii = ((velocities * np.expand_dims(np.sqrt(times), 1)))
        criticalRadii = ((velocities * np.expand_dims(times / np.sqrt(np.log(times)), 1)))
        linearRadii = ((velocities * np.expand_dims(times, 1)))
    #  262 by 150 flattened to 1d array of length 39300
    radii = (np.hstack((sqrtRadii, criticalRadii, linearRadii))).flatten("F")
    ssrwData = np.array([SSRW.flatten("F"), radii, longTs])
    with np.errstate(divide='ignore'):
        SSRWMean = np.log(erfc((ssrwData[1,:]-1)/np.sqrt(ssrwData[2,:]-1))/2)
        ssrwIndices = (ssrwData[1,:] > rMin)
        ax.loglog(-SSRWMean[ssrwIndices], -ssrwData[0,:][ssrwIndices],color='green',linewidth=0.5)
    ax.plot(x,x,color='red')
    os.makedirs(os.path.join(savePath,"integrals"),exist_ok=True)
    fig.savefig(os.path.join(savePath,"integrals","9.png"))


def plotAllMeans(topDir, savePath):
    plt.rcParams.update(
        {'font.size': 10, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    # load in stats
    #path = "/mnt/talapasData/data/pastLine/1000/Line/"
    #savePath = "/home/fransces/Documents/Figures/"
    with open(f"{topDir}variables.json", "r") as v:
        variables = json.load(v)
    tMax = variables['tMax']
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    velocities = np.array(variables['velocities'])
    n = 50
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    sqrtRadii = (velocities * np.expand_dims(np.sqrt(times), 1))
    criticalRadii = (velocities * np.expand_dims(times/np.sqrt(np.log(times)), 1))
    linearRadii = (velocities * np.expand_dims(times, 1))
    # this should be a # oft by 3*# ofvelocities 2d array
    radiiArray = np.hstack((sqrtRadii, criticalRadii, linearRadii))
    x = np.logspace(0,3)

    # past line
    print("past line means!")
    statsLine = np.load(os.path.join(topDir, "LineStats.npy"))  # mean, var, skew, kurtosis
    # mean
    lineFig0, lineAx0 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        gaussianCircleBehavior = radiiArray[:,i]**2 / times
        meanLineBehavior = np.log(erfc(radiiArray[:,i]/(np.sqrt(times))) / 2)
        # lineAx0[0].loglog(gaussianCircleBehavior, - statsLine[0, :, i], '.', ms=1, color=colors[i])
        lineAx0[0].loglog(-meanLineBehavior, - statsLine[0, :, i], '.', ms=1, color=colors[i])
    # lineAx0[0].set_xlabel(r"$r^2/t$")
    lineAx0[0].set_xlabel(r"$-\ln{(\frac{1}{2}\mathrm{erfc}(r/\sqrt{t}))}$")
    # lineAx0[0].set_ylabel(r"$-\langle \ln{(P_{line})} \rangle$")
    lineAx0[0].set_title(r"past line $vt^{1/2}$, v=1e2 (blue) to 2 (yellow)")
    lineAx0[0].plot(x,x,color='red')
    for i in range(50, 100):  # t/sqrt(ln(t))
        gaussianCircleBehavior = radiiArray[:,i]**2 / times
        meanLineBehavior = np.log(erfc(radiiArray[:,i]/np.sqrt(times)))
        # lineAx0[1].loglog(gaussianCircleBehavior, - statsLine[0, :, i], '.', ms=1, color=colors[i - 50])
        lineAx0[1].loglog(-meanLineBehavior, - statsLine[0, :, i], '.', ms=1, color=colors[i - 50])
    # lineAx0[1].set_xlabel(r"$r^2/t$")
    lineAx0[0].set_xlabel(r"$-\ln{(\frac{1}{2}\mathrm{erfc}(r/\sqrt{t}))}$")
    lineAx0[1].set_ylabel(r"$-\langle \ln{(P_{line})} \rangle$")
    lineAx0[1].set_title(r"past line $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 2 (yellow)")
    lineAx0[1].plot(x,x,color='red')
    for i in range(100, 150):  # t
        gaussianCircleBehavior = radiiArray[:,i]**2 / times
        meanLineBehavior = np.log(erfc(radiiArray[:,i]/np.sqrt(times)))
        # lineAx0[2].loglog(gaussianCircleBehavior, - statsLine[0, :, i], '.', ms=1, color=colors[i - 100])
        lineAx0[2].loglog(-meanLineBehavior, - statsLine[0, :, i], '.', ms=1, color=colors[i - 100])
    # lineAx0[2].set_xlabel(r"$r^2/t$")
    lineAx0[0].set_xlabel(r"$-\ln{(\frac{1}{2}\mathrm{erfc}(r/\sqrt{t}))}$")
    lineAx0[2].set_ylabel(r"$- \langle \ln{(P_{line})}\rangle$")
    lineAx0[2].set_title(r"past line $vt$, v=1e-2 (blue) to 2 (yellow)")
    lineAx0[2].plot(x,x,color='red')
    lineFig0.savefig(os.path.join(savePath, "meanLineRegimes30kSystems.png"))


def plotAllScaledVars(topDir, savePath, rMin=3):
    plt.rcParams.update(
        {'font.size': 10, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    # load in stats
    #path = "/mnt/talapasData/data/pastLine/1000/Line/"
    #savePath = "/home/fransces/Documents/Figures/"
    with open(f"{topDir}variables.json", "r") as v:
        variables = json.load(v)
    tMax = variables['tMax']
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    velocities = np.array(variables['velocities'])
    n = 50
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    sqrtRadii = (velocities * np.expand_dims(np.sqrt(times), 1))
    criticalRadii = (velocities * np.expand_dims(times/np.sqrt(np.log(times)), 1))
    linearRadii = (velocities * np.expand_dims(times, 1))
    # this should be a # oft by 3*# ofvelocities 2d array
    radiiArray = np.hstack((sqrtRadii, criticalRadii, linearRadii))
    indices = radiiArray > rMin
    x = np.logspace(0,3)
    expVarX = getExpVarXDotProduct("Dirichlet", [1] * 4)
    # turn into D_ext and D
    D_ext = (1 / 2) * expVarX  # D_ext includes factor of 1/2 by defn.
    D = 1 / 2  # by defn includes factor of 1/2
    lambdaExt = (1 / 2) * D_ext / (D - D_ext)
    # past line
    print("past line means!")
    statsLine = np.load(os.path.join(topDir, "LineStats.npy"))  # mean, var, skew, kurtosis
    # var
    lineFig1, lineAx1 = plt.subplots(3, 1, figsize=(8, 15))
    for i in range(n):  # sqrt
        lineAx1[0].loglog(times[indices[:,i]], statsLine[1, :, i][indices[:,i]]*lambdaExt / (1-velocities[i]**2 * lambdaExt), '.', ms=1, color=colors[i])
    lineAx1[0].set_xlabel(r"$t$")
    lineAx1[0].set_ylabel(r"$\frac{\mathrm{Var}\left[\ln{(P_{line})}\right]\lambda}{1-\lambda v^2}$")
    lineAx1[0].set_title(r"past line $vt^{1/2}$, v=1e-2 (blue) to 2 (yellow), $r>3$")
    for i in range(50, 100):  # t/sqrt(ln(t))
        lineAx1[1].loglog(times[indices[:,i]], statsLine[1, :, i][indices[:,i]]*lambdaExt/ (1-velocities[i-50]**2 * lambdaExt ), '.', ms=1, color=colors[i - 50])
    lineAx1[1].set_xlabel(r"$t$")
    lineAx1[1].set_ylabel(r"$\frac{\mathrm{Var}\left[\ln{(P_{line})}\right]\lambda}{1-\lambda v^2}$")
    lineAx1[1].set_title(r"past line  $vt/\sqrt{\ln{t}}$, v=1e-2 (blue) to 2 (yellow)")
    for i in range(100, 150):  # t
        lineAx1[2].loglog(times[indices[:,i]], statsLine[1, :, i][indices[:,i]]*lambdaExt/(1-velocities[i-100]**2 * lambdaExt ), '.', ms=1, color=colors[i - 100])
    lineAx1[2].set_xlabel(r"$t$")
    lineAx1[2].set_ylabel(r"$\frac{\mathrm{Var}\left[\ln{(P_{line})}\right]\lambda}{1-\lambda v^2}$")
    lineAx1[2].set_title(r"past line $vt$, v=1e-2 (blue) to 2 (yellow)")
    lineFig1.savefig(os.path.join(savePath, "lineVarScaledNewNewDataHindysPrediction.png"))


def plotFinalVarsAndSkews(topDir, savepath):
    stats = np.load(os.path.join(topDir,"LineStats.npy"))
    with open(os.path.join(topDir, "variables.json"), "r") as v:
        variables = json.load(v)
    velocities = np.array(variables['velocities'])
    # var[lnP at tmax] vs v
    fig0, (ax0,ax1) = plt.subplots(2,1,constrained_layout=True,figsize=(4,6))
    ax0.semilogy(velocities, stats[1,-1,:50], '.',color='blue', label="sqrt")
    ax0.semilogy(velocities, stats[1,-1,50:100],'.', color='orange', label=r"moderate")
    ax0.semilogy(velocities,stats[1,-1,100:], '.',color='green', label=r"linear")
    ax0.set_xlabel(r"$v$")
    ax0.set_ylabel(r"$\mathrm{Var}[\ln{P}]$")
    ax0.set_title(r"past line, at tMax=1000")
    ax0.legend()
    # skew[lnP at tmax] vs v
    ax1.semilogy(velocities, stats[2,-1,:50], '.',color='blue', label="sqrt")
    ax1.semilogy(velocities, stats[2,-1,50:100],'.', color='orange', label=r"moderate")
    ax1.semilogy(velocities,stats[2,-1,100:], '.',color='green', label=r"linear")
    ax1.set_xlabel(r"$v$")
    ax1.set_ylabel(r"$\mathrm{Skew}[\ln{P}]$")
    ax1.legend()
    fig0.savefig(os.path.join(savepath,"VarSkewVsVelocityPastLine.png"))


if __name__ == "__main__":
    expVarX = getExpVarXDotProduct("Dirichlet", [1] * 4)
    # turn into D_ext and D
    D_ext = (1 / 4) * expVarX  # D_ext includes factor of 1/2 by defn.
    D = 1 / 4 # by defn includes factor of 1/2
    lambdaExt = (1 / 2) * D_ext / (D - D_ext)

    savePath = "/home/fransces/Documents/Figures/"
    topDir = "/mnt/talapasData/data/pastLineTake3/1000/Line/"
    # topDir = "/mnt/talapasData/data/pastLineTake2/1000/Line/"
    # topDir = "/mnt/talapasData/data/pastLine/1000/Line/"
    # plotDifferentScalings(topDir, savePath, lambdaExt, rMin=2)
    # plotAllScaledVars(topDir, savePath)
    collapseMean(topDir, savePath,rMin=2)
    # plotAllMeans(topDir, savePath)
    # plotFinalVarsAndSkews(topDir, savePath)