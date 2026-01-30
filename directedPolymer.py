import numpy as np
from scipy.stats import skew
from numba import njit, objmode
from matplotlib import pyplot as plt
import sys
import glob
import time

# Terminology:
# "jumpLibrary" are the movements that the polymer can take from one site to the next
# The list of chosen jumps is called "jumps"
# The cumsum of these jumps forms a "walk"
# Every site at every time has a "weight", which has a size x by y by t

# _defaultJumpLibrary = np.array([[0,0], [0,1], [1,0], [1,1]])

# @njit
# def generateSeed():
#     return np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max)

# @njit
# def weight(x, y, t, tMax, randomSeed):
#     index = x*tMax*tMax + y*tMax + t + randomSeed
#     np.random.seed(index)
#     return np.random.randn()

# @njit
# def weightTest(N):
#     arr = np.empty(N)
#     for i in range(N):
#         arr[i] = weight(3,4,5,100,1000)
#     return arr

# @njit
# # steps is a list of tMax steps, starting with [0,0] as the first step
# def createSteps(tMax, jumpLibrary = _defaultJumpLibrary):
#     steps = np.empty((tMax,2), dtype=np.int64)
#     steps[0] = [0,0]
#     for i in range(1,tMax):
#         steps[i] = jumpLibrary[np.random.randint(len(jumpLibrary))]
#     return steps
#     # steps = jumpLibrary[np.random.randint(len(jumpLibrary), size=tMax-1)]
#     # return steps

# @njit
# def stepsToWalk(steps):
#     walk = np.empty(steps.shape, dtype=np.int64)
#     walk[0] = steps[0].copy()
#     for i in range(1, steps.shape[0]):
#         walk[i] = walk[i-1] + steps[i]
#     return walk
#     # return np.cumsum(np.vstack([(0,0),steps]), 0)

# # @njit
# # def computeTotalEnergy(steps, omegas):
# #     walk = stepsToWalk(steps)
# #     energy = 0
# #     for i in range(walk.shape[0]):
# #         energy += omegas[walk[i,0], walk[i,1], i]
# #     return energy
# #     # return np.sum(omegas[walk[:,0], walk[:,1], range(len(walk))])

# @njit
# def computeTotalEnergySeed(steps, randomSeed):
#     walk = stepsToWalk(steps)
#     energy = 0
#     for i in range(walk.shape[0]):
#         energy += weight(walk[i,0], walk[i,1], i, len(steps), randomSeed)
#     return energy
#     # return np.sum(omegas[walk[:,0], walk[:,1], range(len(walk))])

# @njit
# def computeTotalEnergyOmegas(steps, omegas):
#     walk = stepsToWalk(steps)
#     energy = 0
#     for i in range(walk.shape[0]):
#         energy += omegas[walk[i,0], walk[i,1], i]
#     return energy
#     # return np.sum(omegas[walk[:,0], walk[:,1], range(len(walk))])


# @njit
# def proposeMove(tMax, jumpLibrary = _defaultJumpLibrary):
#     # Pick a site t with probability proportional to (t)
#     maxValue = tMax * (tMax-1) // 2
#     # We can map the integer between 1 and maxValue+1 to the underlying integers between 1 and n
#     site = int(np.round(np.sqrt(2*np.random.randint(1, maxValue+1))))
#     return site, jumpLibrary[np.random.randint(len(jumpLibrary))]
#     # Alternative with flat weighting
#     # return np.random.randint(tMax), jumpLibrary[np.random.randint(len(jumpLibrary))]

# # @njit
# def polymerMC(tMax, mcMax, temperature, steps = None, jumpLibrary = _defaultJumpLibrary, precomputeWeights = True):
#     if steps is None:
#         steps = createSteps(tMax, jumpLibrary=jumpLibrary)
    
#     if precomputeWeights:
#         omegas = np.random.randn(tMax, tMax, tMax)
#         energyList = [computeTotalEnergyOmegas(steps, omegas)]
#     else:
#         randomSeed = generateSeed()
#         energyList = [computeTotalEnergySeed(steps, randomSeed)]

#     for _ in range(mcMax-1):
#         t, newDirection = proposeMove(tMax, jumpLibrary = jumpLibrary)
#         oldDirection = steps[t].copy()
#         steps[t] = newDirection
#         if precomputeWeights:        
#             curEnergy = computeTotalEnergyOmegas(steps, omegas)
#         else:        
#             energyList = computeTotalEnergySeed(steps, randomSeed)
#         deltaE = curEnergy - energyList[-1]
#         # Accept the move if it's downhill or with probability exp(-deltaE/temperature)
#         if (deltaE < 0) or (np.random.rand() < np.exp(-deltaE/temperature)):
#             energyList.append(curEnergy)
#         else:
#             energyList.append(energyList[-1])
#             steps[t] = oldDirection.copy()
#     return energyList, steps

# # @njit
# def polymerMC(tMax, mcMax, temperature, steps = None, omegas = None, jumpLibrary = _defaultJumpLibrary, precomputeWeights = True):
#     if steps is None:
#         steps = createSteps(tMax, jumpLibrary=jumpLibrary)
    
#     if omegas is None:
#         omegas = np.random.randn(tMax, tMax, tMax)
#     energyList = [computeTotalEnergyOmegas(steps, omegas)]

#     for _ in range(mcMax-1):
#         t, newDirection = proposeMove(tMax, jumpLibrary = jumpLibrary)
#         oldDirection = steps[t].copy()
#         steps[t] = newDirection
#         curEnergy = computeTotalEnergyOmegas(steps, omegas)
#         deltaE = curEnergy - energyList[-1]
#         # Accept the move if it's downhill or with probability exp(-deltaE/temperature)
#         if (deltaE < 0) or (np.random.rand() < np.exp(-deltaE/temperature)):
#             energyList.append(curEnergy)
#         else:
#             energyList.append(energyList[-1])
#             steps[t] = oldDirection.copy()
#     return energyList, steps

# def meanEDistribution(tMax=100, mcMax=10000, temperature=1, nSystems=100, lowCut=1000):
#     meanE = []
#     for i in range(nSystems):
#         energyList, _ = polymerMC(tMax, mcMax, temperature)
#         meanE.append(np.mean(energyList[lowCut:]))
#         print(i, energyList[-1])
#     return meanE
    

# def skewOfTemp(tempList, tMax=100, mcMax=100000, nSystems=1000, lowCut=1000, jumpLibrary = _defaultJumpLibrary):
#     # Make sure that tempList is in descending order
#     figure = plt.figure(1)
#     tempList[::-1].sort()
#     meanE = np.empty((len(tempList), nSystems))
#     for sysId in range(nSystems):
#         steps = createSteps(tMax, jumpLibrary=jumpLibrary)
#         omegas = np.random.randn(tMax, tMax, tMax)
#         for i, t  in enumerate(tempList):
#             energyList, steps = polymerMC(tMax, mcMax, t, steps = steps, omegas = omegas)
#             meanE[i, sysId] = np.mean(energyList[lowCut:])
#             print(sysId, t, meanE[i,sysId])
#         print(sysId, skew(meanE[:,:sysId], axis=1))
#         figure.clf()
#         ax = figure.add_subplot(111)
#         ax.semilogx(1/tempList, skew(meanE[:,:sysId],axis=1),'o-')
#         ax.set_title(f'systems={sysId}')
#         figure.canvas.draw()
#         figure.canvas.flush_events()
#     return meanE, tempList

# @njit
# def transferMatrix1D(tMax, temperature=0):
#     if temperature == 0:
#         localOptimalEnergy = np.empty(tMax)
#         # The t=0 optimal path starts at the origin
#         localOptimalEnergy[0] = np.random.randn()
#         # print(localOptimalEnergy[0])
#         for t in range(1,tMax):
#             newWeights = np.random.randn(t+1)
#             # print(f'newWeights = {newWeights}')
#             # There's only one path to the largest site so just add the new weight
#             localOptimalEnergy[t] = localOptimalEnergy[t-1] + newWeights[t]
#             for x in range(t-1,0,-1):
#                 localOptimalEnergy[x] = newWeights[x] + np.min(localOptimalEnergy[x-1:x+1])
#             localOptimalEnergy[0] = localOptimalEnergy[0] + newWeights[0]
#             # print(localOptimalEnergy[:t+1])
#             # print(np.min(localOptimalEnergy[:t+1]))
#         return np.min(localOptimalEnergy)
    

# @njit
# def computeWeightedEnergy(partitionFunction, expectedEnergy, x, y):
#     # NOTE: This fails for temperatures that are too small!  If everything is zero then probably we should just take the min or energy?
#     predecessorZ = np.zeros(partitionFunction.shape[2])
#     weightedEnergy = np.zeros(partitionFunction.shape[2])
#     for i in [-1,0]:
#         for j in [-1,0]:
#             weightedEnergy += partitionFunction[x+i,y+j] * expectedEnergy[x+i,y+j]
#             predecessorZ += partitionFunction[x+i,y+j]
#     weightedEnergy /= predecessorZ
#     # if prevBF > 0:
#     #     weightedEnergy /= prevBF
#     # else:
#     #     weightedEnergy = 0
#     return predecessorZ, weightedEnergy

# @njit
# def computeLogPredecessorZ(logZ, x, y):
#     # Find the mean value of logZ for the 4 previous sites, this will be a list of length numTemps
#     # meanLogZ = np.zeros(logZ.shape[2])
#     maxLogZ = np.zeros(logZ.shape[2])
#     predecessorZ = np.zeros(logZ.shape[2])
#     for i in [-1,0]:
#         for j in [-1,0]:
#             for betaIndex in range(logZ.shape[2]):
#                 maxLogZ[betaIndex] = max(maxLogZ[betaIndex],logZ[x+i, y+j, betaIndex])
#             # meanLogZ += logZ[x+i, y+j]/4
#             # maxLogZ = np.max((maxLogZ,logZ[x+i, y+j]), 0)

#     # Shift the max value so that it gets put at the very top of the range
#     maxLogZ -= 700
#     # print(logZ[x,y] - maxLogZ)
#     # meanLogZ = np.mean(logZ[x-1:x+1,y-1:y+1].reshape(4, logZ.shape[2]),0)
#     # We want to return 
#     # np.sum(np.exp(logZ[x-1:x+1, y-1:y+1].reshape(4, logZ.shape[2])),0)
#     # but this runs into precision problems
#     # Instead, factor out the mean value of logZ before taking exponentials
#     for i in [-1,0]:
#         for j in [-1,0]:
#             predecessorZ += np.exp( logZ[x+i,y+j] - maxLogZ )
    
#     return np.log(predecessorZ) + maxLogZ

#     # return np.log(np.sum(np.exp(logZ[x-1:x+1, y-1:y+1].reshape(4, logZ.shape[2]) - meanLogZ),0)) + meanLogZ 

#     # for i in [-1,0]:
#     #     for j in [-1,0]:
#     #         predecessorZ += np.exp(logZ[x+i, y+j])
#     # return np.log(predecessorZ)


@njit
def logSumExp(x):
    a = np.max(x)
    # We're wasting our time if we include entries that are more than np.log(np.finfo(float).eps) = -36.04365338911715 below the max
    cutoff = -37
    expSum = 0
    for val in x:
        # if val-a > cutoff and val != a: # This would eliminate the computation of np.log(np.exp(0)), but it seems to make no difference in the long term.
        if val-a > cutoff:
            expSum += np.exp(val-a)
    return a + np.log(expSum)

    # return a + np.log(np.sum(np.exp(x-a)))

@njit
def transferMatrix2D(tMax, betaList): #, measurementTimes):
    # betaList has to be of length tMax
    assert betaList.shape[0] == tMax, "betaList must be length tMax"
    dataSize = (tMax, tMax)
    logZ = np.full(dataSize, -np.inf).flatten()
    newLogZ = logZ.copy()
    logZ[0] = 0

    # The 4 neighbors coordinates in x and y so that we can work w/ flattened coords
    neighborX = np.array([0,0,-1,-1])
    neighborY = np.array([0,-1,0,-1])
    for t in range(1,tMax):
        weights = np.random.randn(t,t)
        for x in range(0,t):
            indexListX = np.mod(x + neighborX, dataSize[0]) * dataSize[1]
            for y in range(0,t):
                indexListY = indexListX + np.mod(y + neighborY, dataSize[1])
                newLogZ[x * dataSize[1] + y] = -weights[x,y] * betaList[t] + logSumExp(logZ[indexListY])
        # replace the current values with the new values
        logZ, newLogZ = newLogZ, logZ
        yield logZ, t

@njit
def measurePartitionFunction(logZ, t, tMax):
    pointToPlane = logSumExp(logZ)
    # Pick the halfway line
    half = t//2
    pointToLine = logSumExp(logZ[half * tMax : half * tMax + t])
    # Pick the center point
    pointToPoint = logZ[half * tMax + half]
    return pointToPlane, pointToLine, pointToPoint

def readLogZFiles(globString):
    all = []
    for f in glob.glob(globString):
        all.append(np.loadtxt(f))
    return np.array(np.vstack(all))

if __name__ == "__main__":

    # Call as `python3 directedPolymer.py tMax numSystems outFile betaString`
    inputIndex = 1
    tMax = int(sys.argv[inputIndex]); inputIndex += 1
    numSystems = int(sys.argv[inputIndex]); inputIndex += 1
    outFileName = sys.argv[inputIndex]; inputIndex += 1
    betaString = sys.argv[inputIndex]; inputIndex += 1 # Example string, "np.ones(tMax)", 
    
    measurementTimes = np.geomspace(1,tMax, 2 * np.log10(tMax).astype(int) + 1).astype(int)
    # make tMax one more so that our measurements end at the input value, rather than input value-1
    tMax += 1
    beta = eval(betaString)
    beta0List = np.geomspace(.1,10,9)
    s = time.time()
    with open(outFileName, 'a') as file:        
        for _ in range(numSystems):
            for beta0 in beta0List:
                measureIndex = 0
                for logZ, t in transferMatrix2D(tMax, beta0*beta):
                    # Make measurements that are log-spaced
                    if measurementTimes[measureIndex] == t:
                        measureIndex += 1
                        p2Plane, p2Line, p2Point = measurePartitionFunction(logZ, t, tMax)
                        # measurementTimes = np.delete(measurementTimes, 0)
                        file.write(f'{t}, {beta0}, {p2Plane}, {p2Line}, {p2Point} \n')
                        # print(t, time.time()-s, p2Plane, p2Line, p2Point)


    # print(logScaling)
    # temp0 = np.geomspace(tempMin, tempMax, numTemp)
    # if logScaling:
    #     tempList = np.multiply.outer(temp0, np.sqrt( np.log( np.e * np.arange(1,tMax+1) ) ) )
    # elif sqrtScaling:
    #     tempList = np.multiply.outer(temp0, np.sqrt(np.arange(1,tMax+1)))
    # else:
    #     tempList = np.multiply.outer(temp0, np.ones(tMax) )
        
    # for sysId in range(numSystems):
    #     logZ = transferMatrix2D(tMax, tempList)
    #     # Format things so that they save as a row, rather than a column
    #     pointToPlane = np.array([logSumPartitionFunction(logZ[:,:,i]) for i in range(numTemp)]).reshape(-1,1).T
    #     with open(outFile, 'a') as file:
    #         np.savetxt(file, pointToPlane)
    #     print(sysId)
    # # for i in range(numTemp):
    # #     print(f'Temp={tempList[i]}, logZ = {logSumPartitionFunction(logZ[:,:,i])}')
