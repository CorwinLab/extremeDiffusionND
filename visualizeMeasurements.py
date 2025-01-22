from sympy.physics.units import velocity

import evolve2DLattice as ev
import memEfficientEvolve2DLattice as m
import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm
import os
import time
import math

def generateGifRWRE(occupancy, maxT, alphas, startT=1, pathName, listOfTimes=None):
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

def getLastValsLog(path,velocityIdx,savePath):
    """
    path: str; specify path to data
    velocity; int; needs to be index of the velocity at which you want to probe
    scaling; int; 0 = linear, 1 = sqrt, 2= vt/ln(t); 3 = vt/sqrt(ln(t)); 4 = fixed radius
    savePath: str; specify path to which pngs saved
    """
    info = np.load(f"{path}/info.npz")
    times = info['times']
    velocities = info['velocities'].flatten()
    os.makedirs(os.path.join(savePath,"distribution",f"v{velocities[velocityIdx]}"),exist_ok=True)
    os.makedirs(os.path.join(savePath,"cumulative"),exist_ok=True)
    saveFile = os.path.join(savePath,"distribution",f"v{velocities[velocityIdx]}","lastValsLog")
    # cumulPath = os.path.join(savePath,"cumulative")
    x = np.linspace(-4,4,50000)
    p = norm.pdf(x,0,1)
    c = norm.cdf(x,0,1)
    # only get data files
    fileList = glob.glob("*.npy",root_dir=path)
    # print(f"time for glob: {time.time()-s}")
    lastVT = []
    lastVSqrtT = []
    lastVTOnSqrtLogT = []
    # s = time.time()
    for file in fileList:
        data = np.load(f"{path}/{file}")
        # TODO: implement thing so it doesn't just grab
        #  the last val. as a proxy for maxT...
        valsAtVelocityVT = data[0,:,velocityIdx]
        valsAtVelocityVSqrtT = data[1,:,velocityIdx]
        valsAtVelocityVTOnSqrtLogT = data[3,:,velocityIdx]
        lastVT.append(valsAtVelocityVT[-1])
        lastVSqrtT.append(valsAtVelocityVSqrtT[-1])
        lastVTOnSqrtLogT.append([-1])
        # lastVT.append(valsAtVelocityVT[np.max(np.where(valsAtVelocityVT != 0))])
        # lastVSqrtT.append(valsAtVelocityVSqrtT[np.max(np.where(valsAtVelocityVSqrtT != 0))])
        # lastVTOnSqrtLogT.append([np.max(np.where(valsAtVelocityVTOnSqrtLogT != 0))])
        # logProbVT.append(np.log(valsAtVelocityVT[np.max(np.where(~np.isnan(valsAtVelocityVT)))]))
        # logProbVSqrtT.append(np.log(valsAtVelocityVSqrtT[np.max(np.where(~np.isnan(valsAtVelocityVSqrtT)))]))
        # logProbCrit.append(np.log(valsAtVelocityCrit[np.max(np.where(~np.isnan(valsAtVelocityCrit)))]))
    np.savez_compressed(saveFile,VT=lastVT,VSqrtT=lastVSqrtT,VTOnSqrtLogT=lastVTOnSqrtLogT)

def getHistogram(histogramPath):
    lastLogVals = np.load(histogramPath)
    VT = lastLogVals['VT']
    VSqrtT = lastLogVals['VSqrtT']
    VTOnSqrtLogT = lastLogVals['VTOnSqrtLogT']
    x = np.linspace(-3,3,50000)
    p = norm.pdf(x,0,1)
    c = norm.cdf(x,0,1)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(5,4),dpi=150)
    ax1.set_yscale("log")
    # # print(f"time to do entire for loop: {time.time() - s}")
    # # s = time.time()
    # mean, var = np.mean(logProb), np.var(logProb)
    # #print(f"time it takes to do mean var calc: {time.time()-s}")
    # plt.figure(figsize=(5,4),constrained_layout=True,dpi=150)
    # plt.yscale("log")
    # plt.hist((logProb-mean)/np.sqrt(var),bins=500,density=True)
    # plt.semilogy(x,p,color='black')
    # # plt.title(f"centered and scaled histogram of ln(prob) "
    # #          f"\n at t={t} with v={velocity} \n from {path} "
    # #          f"\n with gaussian in black")
    # plt.savefig(saveNameDistribution)
    # plt.figure(figsize=(5,4),constrained_layout=True,dpi=150)
    # plt.plot(np.sort((logProb-mean)/np.sqrt(var)),np.arange(len(logProb))/len(logProb),label="RWRE")
    # plt.plot(x,c,color='black',label="Normal")
    # #plt.title(f"cdf of ln(prob) at t={t} with velocity={velocity}"
    # #          f"\n from {path}"
    # #          f"\n with gaussian in black")
    # plt.savefig(saveNameCumul)



# TODO: fix because switched format
# to get histograms of ln(probs)
def visualizeDistributionOld(path,velocity,t,savePath):
    if not os.path.isdir(savePath):
        os.makedirs(savePath)
    x = np.linspace(-4,4,50000)
    p = norm.pdf(x,0,1)
    c = norm.cdf(x,0,1)
    s = time.time()
    filelist = glob.glob("sphere*",root_dir=path)
    print(f"time for glob: {time.time()-s}")
    logProb = []
    s = time.time()
    for file in filelist:
        data = pd.read_csv(f"{path}/{file}")
        # TODO: implement thing so it doesn't just grab
        #  the last val. as a proxy for maxT...
        val = np.array(data[velocity])[-1]
        logProb.append(np.log(val))
    print(f"time to do entire for loop: {time.time() - s}")
    s = time.time()
    mean, var = np.mean(logProb), np.var(logProb)
    print(f"time it takes to do mean var calc: {time.time()-s}")
    plt.figure(figsize=[9.6,7.2])
    plt.yscale("log")
    plt.hist((logProb-mean)/np.sqrt(var),bins=500,density=True)
    plt.semilogy(x,p,color='black')
    plt.title(f"centered and scaled histogram of ln(prob) "
              f"\n at t={t} with v={velocity} \n from {path} "
              f"\n with gaussian in black")
    plt.savefig(os.path.join(savePath,"vel"+velocity,"distribution.png"),dpi=150)
    plt.figure(figsize=[9.6,7.2])
    plt.plot(np.sort((logProb-mean)/np.sqrt(var)),np.arange(len(logProb))/len(logProb))
    plt.plot(x,c,color='black')
    plt.title(f"cdf of ln(prob) at t={t} with velocity={velocity}"
              f"\n from {path}"
              f"\n with gaussian in black")
    plt.savefig(os.path.join(savePath,"vel"+velocity,"cdf.png"),dip=150)
    np.save(os.path.join(savePath,"lastProbs.txt"),logProb)

# figs for comps?
# plt.figure(figsize=(5,4),constrained_layout=True,dpi=150)
# n =
# colors = plt.cm.jeet(np.linspace(0,1,n))
# for i in range(n):
#       plt.loglog(time[2<(vels[i]*time/(np.sqrt(np.log(time))))], varLog[3,:,i][2<(vels[i]*time/(np.sqrt(np.log(time))))], color=colors[i])
# plt.legend(vels)
# plt.xlabel("Time");plt.ylabel(r"Var($\log{P}$)")
# plt.savefig("/home/fransces/Documents/code/extremeDiffusionND/compFigs/titlehere.png")