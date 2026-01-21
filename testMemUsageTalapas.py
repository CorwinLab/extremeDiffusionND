import numpy as np
import dynamicRangeEvolve2DLattice as d

if __name__ == "__main__":
    L = 10000
    tMax = 10000
    startT = 9000
    logOcc = np.full((2*L+1, 2*L+1), -np.inf)
    logOcc[11000:19000, 11000:19000] = np.random.rand(18000,18000)
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    velocities = np.concatenate((np.linspace(0.1,0.6,11),np.linspace(0.61,0.99,39),np.linspace(0.991,1,10)))
    saveInterval = 3
    logOccFileName = "/scratch/jamming/franssces/data/numbaHistograms/10000/memTest.npy"
    logOccTimeFileName = "/scratch/jamming/franssces/data/numbaHistograms/10000/timeMemTest.npy"
    cumLogProbFileName = "/projects/jamming/franssces/data/numbaHistograms/10000/memTest.npy"
    finalCumLogProbFileName = "/projects/jamming/franssces/data/numbaHistograms/10000/finalMemTest.npy"
    radiiSqArray = (velocities * np.expand_dims(times, 1))**2
    cumLogProbList = []

    d.evolveAndMeasure(logOccFileName, logOccTimeFileName, cumLogProbFileName,
                       finalCumLogProbFileName, cumLogProbList, logOcc,
                       radiiSqArray, times, saveInterval, startT)