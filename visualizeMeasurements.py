import evolve2DLattice as ev
import glob
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import norm
import os
import time

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
