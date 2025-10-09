import numpy as np
from scipy.stats import skew
from numba import njit
from matplotlib import pyplot as plt

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
def weightTest(N):
    arr = np.empty(N)
    for i in range(N):
        arr[i] = weight(3,4,5,100,1000)
    return arr

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
def computeTotalEnergySeed(steps, randomSeed):
    walk = stepsToWalk(steps)
    energy = 0
    for i in range(walk.shape[0]):
        energy += weight(walk[i,0], walk[i,1], i, len(steps), randomSeed)
    return energy
    # return np.sum(omegas[walk[:,0], walk[:,1], range(len(walk))])

@njit
def computeTotalEnergyOmegas(steps, omegas):
    walk = stepsToWalk(steps)
    energy = 0
    for i in range(walk.shape[0]):
        energy += omegas[walk[i,0], walk[i,1], i]
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
    # return np.random.randint(tMax), jumpLibrary[np.random.randint(len(jumpLibrary))]

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

# @njit
def polymerMC(tMax, mcMax, temperature, steps = None, omegas = None, jumpLibrary = _defaultJumpLibrary, precomputeWeights = True):
    if steps is None:
        steps = createSteps(tMax, jumpLibrary=jumpLibrary)
    
    if omegas is None:
        omegas = np.random.randn(tMax, tMax, tMax)
    energyList = [computeTotalEnergyOmegas(steps, omegas)]

    for _ in range(mcMax-1):
        t, newDirection = proposeMove(tMax, jumpLibrary = jumpLibrary)
        oldDirection = steps[t].copy()
        steps[t] = newDirection
        curEnergy = computeTotalEnergyOmegas(steps, omegas)
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
    

def skewOfTemp(tempList, tMax=100, mcMax=100000, nSystems=1000, lowCut=1000, jumpLibrary = _defaultJumpLibrary):
    # Make sure that tempList is in descending order
    figure = plt.figure(1)
    tempList[::-1].sort()
    meanE = np.empty((len(tempList), nSystems))
    for sysId in range(nSystems):
        steps = createSteps(tMax, jumpLibrary=jumpLibrary)
        omegas = np.random.randn(tMax, tMax, tMax)
        for i, t  in enumerate(tempList):
            energyList, steps = polymerMC(tMax, mcMax, t, steps = steps, omegas = omegas)
            meanE[i, sysId] = np.mean(energyList[lowCut:])
            print(sysId, t, meanE[i,sysId])
        print(sysId, skew(meanE[:,:sysId], axis=1))
        figure.clf()
        ax = figure.add_subplot(111)
        ax.semilogx(1/tempList, skew(meanE[:,:sysId],axis=1),'o-')
        ax.set_title(f'systems={sysId}')
        figure.canvas.draw()
        figure.canvas.flush_events()
    return meanE, tempList

@njit
def transferMatrix1D(tMax, temperature=0):
    if temperature == 0:
        localOptimalEnergy = np.empty(tMax)
        # The t=0 optimal path starts at the origin
        localOptimalEnergy[0] = np.random.randn()
        # print(localOptimalEnergy[0])
        for t in range(1,tMax):
            newWeights = np.random.randn(t+1)
            # print(f'newWeights = {newWeights}')
            # There's only one path to the largest site so just add the new weight
            localOptimalEnergy[t] = localOptimalEnergy[t-1] + newWeights[t]
            for x in range(t-1,0,-1):
                localOptimalEnergy[x] = newWeights[x] + np.min(localOptimalEnergy[x-1:x+1])
            localOptimalEnergy[0] = localOptimalEnergy[0] + newWeights[0]
            # print(localOptimalEnergy[:t+1])
            # print(np.min(localOptimalEnergy[:t+1]))
        return np.min(localOptimalEnergy)
    

@njit
def computeWeightedEnergy(boltzmannFactor, expectedEnergy, x, y):
    prevBF = 0
    weightedEnergy = 0
    for i in [-1,0]:
        for j in [-1,0]:
            weightedEnergy += boltzmannFactor[x+i,y+j] * expectedEnergy[x+i,y+j]
            prevBF += boltzmannFactor[x+i,y+j]
    if prevBF > 0:
        weightedEnergy /= prevBF
    else:
        weightedEnergy = 0
    return prevBF, weightedEnergy

@njit
def transferMatrix2D(tMax, temp):
    expectedEnergy = np.zeros((tMax, tMax))
    newExpectedEnergy = np.zeros((tMax, tMax))
    boltzmannFactor = np.zeros((tMax, tMax))
    newBoltzmannFactor = np.zeros((tMax, tMax))
    # The t=0 optimal path starts at the origin
    expectedEnergy[0,0] = np.random.randn()
    boltzmannFactor[0,0] = 1

    for t in range(1,tMax):
        newWeights = np.random.randn(t+1,t+1)
        for x in range(0,t):
            for y in range(0,t):
                prevBF, weightedEnergy = computeWeightedEnergy(boltzmannFactor, expectedEnergy, x, y)
                # This feels sloppy, but it relies on the fact that the boltzmann factor with negative -1 index will be zero                
                newBoltzmannFactor[x,y] = np.exp(-newWeights[x,y]/temp) * prevBF
                newExpectedEnergy[x,y] = newWeights[x,y] + weightedEnergy
        # Put the new values into the regular values
        expectedEnergy, newExpectedEnergy = newExpectedEnergy, expectedEnergy
        boltzmannFactor, newBoltzmannFactor = newBoltzmannFactor, boltzmannFactor
        # Normalize the boltzmannFactor so that it's a probability
        boltzmannFactor /= np.sum(boltzmannFactor)
    return expectedEnergy, boltzmannFactor