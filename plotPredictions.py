from visualizeMeasurements import plotMasterCurve
import dataAnalysis as d
import numpy as np
from matplotlib import pyplot as plt
from visualizeMeasurements import colorsForLambda
import json

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
    #
    statsFileList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]
    savePath = "/home/fransces/Documents/Figures/Paper/constantCollapse.pdf"
    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])
    # tMaxList = np.flip(np.array([2, 10, 37, 100, 402, 639, 4047, 9816]))

    # should give me x and y, with data chopped off
    scalingFunc, vlp, vs, times, ls = d.prepLossFunc(statsFileList, tMaxList,
                                                     vlpMax=1e-3,alpha=1)
    g = vlp / (ls * vs**2)
    print(f"mean g, std g: {np.mean(g), np.std(g)}")
    alphas = np.linspace(-2,2,1001)
    s = [np.std(g - times**alpha) for alpha in alphas]
    tedge = np.geomspace(10, 1e4)
    binnedMedianG = [np.median(g[(times > tedge[i]) * (times < tedge[i + 1])]) for i in range(len(tedge) - 1)]


    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    plt.ion()
    # markers = ['o'] * 7 + ['D'] + ['v'] + ['s']
    # this will run through all your files once to pull out the
    # list of lambdas. the order should correspond to the filelist
    expVarXList, lambdaExtVals = d.getListOfLambdas(statsFileList)
    colors = colorsForLambda(lambdaExtVals)
    plt.figure(figsize=(5, 5), constrained_layout=True, dpi=150)
    # plt.gca().set_aspect('equal')
    # this will plot the data
    # colors = ['darkgreen', 'dodgerblue', 'darkslateblue']
    # tGrid = np.arange(1,1e4)
    tGrid = [0,1e5]
    for i in range(len(statsFileList)):
        # data
        file = statsFileList[i]
        print(f"{file}")
        # label: distribution name
        tempData, label = d.processStats(file)
        vlp = tempData[0,:]  # var[ln[P(r(t))]]
        r = tempData[1,:]  # radii
        t = tempData[2,:]  # time
        l = tempData[3,0]  # lambda_ext
        indices = (r > 2)
        # velocities
        vLin = r / t**(1)
        # vMod = r * np.sqrt(np.log(t)) / t
        # vSqrt = r / np.sqrt(t)
        # predictions
        #linPred = (l) * np.ones_like(tGrid)

        # modPred = (l/2) * (1 / np.log(tGrid))
        # sqrtPred = (l/2) * (1/ tGrid)
        # plot data
        # plt.loglog(t[indices], vlp[indices] / vLin[indices]**2, marker='.',color=colors[i], alpha=0.1)
        plt.plot(t[indices], (1/(l*vLin[indices]**2))*vlp[indices],'.',color=colors[i],alpha=.05, zorder=np.random.rand())

        # plt.loglog(t[indices], vlp[indices] / vMod[indices]**2, marker=markers[i],color=colors[1], alpha=0.1)
        # plt.loglog(t[indices], vlp[indices] / vSqrt[indices]**2, marker=markers[i],color=colors[2], alpha=0.1)
        # plot predictions
        # plt.loglog(tGrid, modPred, color='k',linestyle='dotted')
        # plt.loglog(tGrid, sqrtPred, color='k', linestyle='dashdot')
    # collapssed prediction

    # linPred2 = (2/3)*np.ones_like(tGrid)  # this should be whatever value c is; for now 1
    # plt.semilogx(tGrid,linPred2, color='k',linestyle='dashed',zorder=100000)
    plt.semilogx(tedge[1:], binnedMedianG, label="binned median g",color='black')
    plt.ylim([0,4/3])
    plt.xlim([1,2e4])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\frac{\displaystyle 1}{\lambda_{\mathrm{ext}}v^2}\mathrm{Var}_\nu \left[\ln{\left(\mathbb{P}^{\bm{\xi}}\left(|\vec{S}(t)|>r(t)\right)\right)}\right]$")

    plt.savefig(f"{savePath}")