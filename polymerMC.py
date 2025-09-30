import numpy as np
from scipy.stats import skew
from numba import njit

# Terminology:
# "jumpLibrary" are the movements that the polymer can take from one site to the next
# The list of chosen jumps is called "jumps"
# The cumsum of these jumps forms a "walk"
# Every site at every time has a "weight", which has a size x by y by t

_defaultJumpLibrary = np.array([[0,0], [0,1], [1,0], [1,1]])

@njit
def generateSeed():
    return np.random.randint(np.iinfo(np.int64).min, np.iinfo(np.int64).max)

@njit
def weight(x, y, t, tMax, randomSeed):
    index = x*tMax*tMax + y*tMax + t + randomSeed
    np.random.seed(index)
    return np.random.randn()

@njit
# steps is a list of tMax steps, starting with [0,0] as the first step
def createSteps(tMax, jumpLibrary = _defaultJumpLibrary):
    steps = np.empty((tMax,2), dtype=np.int64)
    steps[0] = [0,0]
    for i in range(1,tMax):
        steps[i] = jumpLibrary[np.random.randint(len(jumpLibrary))]
    return steps
    # steps = jumpLibrary[np.random.randint(len(jumpLibrary), size=tMax-1)]
    # return steps

@njit
def stepsToWalk(steps):
    walk = np.empty(steps.shape, dtype=np.int64)
    walk[0] = steps[0].copy()
    for i in range(1, steps.shape[0]):
        walk[i] = walk[i-1] + steps[i]
    return walk
    # return np.cumsum(np.vstack([(0,0),steps]), 0)

# @njit
# def computeTotalEnergy(steps, omegas):
#     walk = stepsToWalk(steps)
#     energy = 0
#     for i in range(walk.shape[0]):
#         energy += omegas[walk[i,0], walk[i,1], i]
#     return energy
#     # return np.sum(omegas[walk[:,0], walk[:,1], range(len(walk))])

@njit
def computeTotalEnergy(steps, randomSeed):
    walk = stepsToWalk(steps)
    energy = 0
    for i in range(walk.shape[0]):
        energy += weight(walk[i,0], walk[i,1], i, len(steps), randomSeed)
    return energy
    # return np.sum(omegas[walk[:,0], walk[:,1], range(len(walk))])


@njit
def proposeMove(tMax, jumpLibrary = _defaultJumpLibrary):
    # Pick a site t with probability proportional to (t)
    maxValue = tMax * (tMax-1) // 2
    # We can map the integer between 1 and maxValue+1 to the underlying integers between 1 and n
    site = int(np.round(np.sqrt(2*np.random.randint(1, maxValue+1))))
    return site, jumpLibrary[np.random.randint(len(jumpLibrary))]
    # Alternative with flat weighting
    # return np.random.randint(tMax-1), jumpLibrary[np.random.randint(len(jumpLibrary))]

# @njit
def polymerMC(tMax, mcMax, temperature, jumpLibrary = _defaultJumpLibrary):
    randomSeed = generateSeed()
    print('generatedSeed')
    # omegas = np.random.randn(tMax, tMax, tMax)
    steps = createSteps(tMax, jumpLibrary=jumpLibrary)
    print('generatedSteps')
    energyList = [computeTotalEnergy(steps, randomSeed)]
    print('generatedEnergy')

    for _ in range(mcMax-1):
        t, newDirection = proposeMove(tMax, jumpLibrary = jumpLibrary)
        oldDirection = steps[t].copy()
        steps[t] = newDirection
        curEnergy = computeTotalEnergy(steps, randomSeed)
        deltaE = curEnergy - energyList[-1]
        # Accept the move if it's downhill or with probability exp(-deltaE/temperature)
        if (deltaE < 0) or (np.random.rand() < np.exp(-deltaE/temperature)):
            energyList.append(curEnergy)
        else:
            energyList.append(energyList[-1])
            steps[t] = oldDirection.copy()
    return energyList, steps

def meanEDistribution(tMax=100, mcMax=10000, temperature=1, nSystems=100, lowCut=1000):
    meanE = []
    for i in range(nSystems):
        energyList, _ = polymerMC(tMax, mcMax, temperature)
        meanE.append(np.mean(energyList[lowCut:]))
        print(i, energyList[-1])
    return meanE
    

def skewOfTemp(tempList, tMax=100, mcMax=100000, nSystems=1000, lowCut=1000):
    meanE = np.empty((len(tempList), nSystems))
    for sysId in range(nSystems):
        for i, t  in enumerate(tempList):
            energyList, _ = polymerMC(tMax, mcMax, t)
            meanE[i, sysId] = np.mean(energyList[lowCut:])
            print(sysId, t, meanE[i,sysId])
        print(skew(meanE[:,:sysId], axis=1))
    return meanE