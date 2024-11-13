import evolve2DLattice as ev
import memEfficientEvolve2DLattice as m
import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm
import os
import time

def generateGifRWRE(occupancy, maxT, alphas, startT,listOfTimes=None):
    os.makedirs("/home/fransces/Documents/code/RWREgifnew/",exist_ok=True)
    if listOfTimes is None:
        listOfTimes = np.unique(np.geomspace(startT, maxT, num=10).astype(int))
    for t in range(startT, maxT):
        occupancy = m.updateOccupancy(occupancy, t, alphas)
        if t in listOfTimes:
            a = np.copy(occupancy)
            #a[np.isinf(a)] = np.nan
            plt.imshow(a)
            plt.savefig(f'/home/fransces/Documents/code/RWREgifnew/{t}.png',bbox_inches='tight')
    #return listOfTimes, np.array(PDFS)


def generateGifSSRW(occupancy, maxT, distribution, params, isPDF,boundary):
    os.makedirs("/home/fransces/Documents/code/SSRWgifnew/",exist_ok=True)
    listoftimes = np.unique(np.geomspace(1,maxT,num=10).astype(int))
    for t, occ in ev.evolve2DLattice(occupancy, maxT, distribution, params, isPDF,boundary=boundary):
        if t in listoftimes:
            a = np.copy(occ)
            plt.imshow(a)
            plt.savefig(f"/home/fransces/Documents/code/SSRWgif/{t}.png",bbox_inches='tight')

def collapseAlphaAndVar(paths,alphaList, scaling=3):
    """
    assumes stats.npz files are already created

    paths: list of strs defining where to get data
    alphaList: list of floats defining the alphas; order corresponds to paths
    scaling: int (0-3) corresponding to scaling vt, vt^1/2, vt/ln(t), vt/sqrt(ln(t)
        defaults to vt/sqrt(ln(t))

    returns plot with expected scaling of var(ln(prob) / (v^2 / (2pi^3*alpha))
    """
    plt.figure(figsize=[9.6, 7.2])
    plt.yscale("log")
    plt.xscale("log")
    # assumes /data/memwhatever/dirichlet/ALPHAsomething/L5000/tMax10000
    for i in len(paths):
        info = np.load(f"{paths[i]}/info.npz")  # times, velocities..
        times = info['times']
        velocities = info['velocities'].flatten()
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



# to get histograms of ln(probs)
def visualizeDistribution(path,velocity,t,savePath):
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
