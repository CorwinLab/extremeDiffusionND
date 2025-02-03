from lDiffusionLink import libDiffusion
import numpy as np

if __name__ == '__main__':
    L = 3
    d = libDiffusion.DiffusionND([1, 1, 1, 1], 3)
    time =  50
    xvals = np.arange(-L, L+1)
    xx, yy = np.meshgrid(xvals, xvals)
    distArray = np.sqrt(xx ** 2 + yy ** 2)
    
    for t in range(time):
        R = 1
        d.iterateTimestep()
        print(d.logIntegratedProbability([[R]]))
        