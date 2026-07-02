import numpy as np
import scipy.stats # import skew
import scipy.special # import logsumexp
from numba import njit, vectorize, int64, objmode
import matplotlib.pyplot as plt
#from scipy.stats import skew
#from matplotlib import pyplot as plt
import sys
import glob
#import time

# from parfor import parfor

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
def previousNeighborVectors(dim):
    unitVectors = -np.eye(dim).astype(int64)
    # There will be 2^dim neighbor vectors
    vectorList = np.zeros((2**dim, dim)).astype(int64)
    for i in range(2**dim):
        for pos in range(dim):
            # Turn i into a binary mask and use that to add up the terms of the unitvectors
            if (i >> pos) & 1:                
                vectorList[i,:] += unitVectors[pos]
    return vectorList


# TODO: rewrite this to work in arbitrary dimensions as there's some indication that
# there may only be a true phase transition for d > 2

# Take in a single index and return a list of indices into the multidimensional array
@njit
def unravelIndex(index, shape):
    dim = len(shape)
    indices = np.empty(dim, dtype=np.int64)
    for d in range(dim):
        indices[dim-1-d] = index % shape[dim-1-d]
        index //= shape[dim-1-d]
    return indices

# Take in a list of indices into the multidimensional array and return a single index
# Requires that both indices and shape be numpy arrays
@njit
def ravelIndex(indices, shape):
    dim = len(shape)
    index = 0
    for d in range(dim):
        index += indices[d]*np.prod(shape[d:-1])
        # print(index)
    return index

@njit
def transferMatrixND(dim, tMax, betaList): #, measurementTimes):
    # betaList has to be of length tMax
    assert betaList.shape[0] == tMax, "betaList must be length tMax"
    dataShape = np.array([tMax]*dim)
    # Create a flattened version of the logZ data
    logZ = np.full(tMax**dim, -np.inf)
    newLogZ = logZ.copy()

    # Create a list of all the previous neighbor directions
    prevNeighbors = previousNeighborVectors(dim)

    # Treat the t=0 case separately so we don't have to deal with the missing previous values, etc
    logZ[0] = np.random.randn()*betaList[0]
    yield logZ, 0

    for t in range(1,tMax):
        # Weights will fill in every location up to t+1 in every dimension
        weights = np.random.randn((t+1)**dim)
        # We need to iterate through every location for which x < t+1, y < t+1, z < t+1, etc.  There will be a total of (t+1)*dim locations
        localShape = np.array([t+1]*dim)
        for site in range(0, (t+1)**dim):
            # Convert the 1d site index into multidimensional indices
            indices = unravelIndex(site, localShape)
            # print(f'indices ={indices}, {indices.shape}')
            # Loop through all previous neighbors
            nIndex = np.empty(2**dim).astype(int64)
            for i, nVec in enumerate(prevNeighbors):
                # Find the index of the previous neighbor
                nIndex[i] = ravelIndex(np.mod(indices + nVec, dataShape), dataShape)
            # Convert the indices associated w/ site into a 1d index for dataShape
            globalSite = ravelIndex(indices, dataShape)
            # Set the weight for the site
            # newLogZ[globalSite] = logSumExp(logZ[nIndex])
            newLogZ[globalSite] = -weights[site] * betaList[t] + logSumExp(logZ[nIndex])

        logZ, newLogZ = newLogZ, logZ
        yield logZ, t


@njit
def transferMatrix2D(tMax, betaList): #, measurementTimes):
    # betaList has to be of length tMax
    assert betaList.shape[0] == tMax, "betaList must be length tMax"
    dataSize = (tMax, tMax)
    logZ = np.full(dataSize, -np.inf).flatten()
    newLogZ = logZ.copy()

    # The 4 neighbors coordinates in x and y so that we can work w/ flattened coords
    neighborX = np.array([0,0,-1,-1])
    neighborY = np.array([0,-1,0,-1])

    # Treat the t=0 case separately so we don't have to deal with the missing previous values, etc
    logZ[0] = np.random.randn()*betaList[0]
    yield logZ, 0

    for t in range(1,tMax):
        weights = np.random.randn(t+1,t+1)
        # print(f'weights={weights}')
        for x in range(0,t+1):
            # print(f'x={x}')
            indexListX = np.mod(x + neighborX, dataSize[0]) * dataSize[1]
            for y in range(0,t+1):
                # print(f'y={y}')
                indexListY = indexListX + np.mod(y + neighborY, dataSize[1])
                # print(f'indexListY={indexListY}')
                newLogZ[x * dataSize[1] + y] = -weights[x,y] * betaList[t] + logSumExp(logZ[indexListY])
                # print(x * dataSize[1] + y, logSumExp(logZ[indexListY]))
        # replace the current values with the new values
        logZ, newLogZ = newLogZ, logZ
        yield logZ, t

@njit
def measurePartitionFunction(logZ, tMax, dim=2):
    # Return measurements of the origin, line through origin, plane through origin, etc
    measurement = np.empty(dim+1)
    for d in range(dim+1):
        measurement[d] = logSumExp(logZ[:tMax**d])
    return measurement

    # pointToPlane = logSumExp(logZ)
    # # For point to line let's report the x=0 line
    # pointToLine = logSumExp(logZ[:tMax])
    # # half = t//2
    # # pointToLine = logSumExp(logZ[half * tMax : half * tMax + t])
    
    # # pointToPoint = logZ[half * tMax + half]
    # # Pick the origin point
    # pointToPoint = logZ[0]
    # return pointToPlane, pointToLine, pointToPoint

def readLogZFiles(globString):
    files = glob.glob(globString)
    maxT = [f.split('/')[-1].split(',')[0].split('=')[1] for f in files]
    maxT = np.array(maxT).astype(int)
    betaList = [f.split(',')[1].split('=')[1][:-4] for f in files]
    betaList = np.array(betaList).astype(float)
    s = np.argsort(betaList)
    betaList = betaList[s]
    maxT = maxT[s]
    # Run through each
    # meanLog, varLog, skewLog = [], [], []
    meanF, varF, skewF = [], [], []
    logMeanZ, logVarZ, logSkewZ = [], [], []
    for index, f in enumerate(np.array(files)[s]):
        print(f)
        a = np.loadtxt(f, delimiter=',')

        # The data stored in the files is ln(Z).  To turn this into a free energy we use
        # F = - ln(Z) / beta.
        meanF.append(- np.mean(a,0) / betaList[index])
        varF.append(np.var(a,0) / betaList[index]**2)
        skewF.append(scipy.stats.skew(a))

        # The data stored in the files is ln(Z).  To turn this into ln<Z> we use
        # ln<Z> = ln( sum(exp(ln(Z)))/N )  = ln(sum(exp(Z))) - ln(N)
        logMoment1 = scipy.special.logsumexp(a, 0) - np.log(a.shape[0])
        logMoment2 = scipy.special.logsumexp(2*a, 0) - np.log(a.shape[0])
        logMoment3 = scipy.special.logsumexp(3*a, 0) - np.log(a.shape[0])
        logMeanZ.append(logMoment1)
        logVarZ.append(scipy.special.logsumexp(np.vstack([logMoment2, 2*logMoment1]), 0, b = [[1]*a.shape[1],[-1]*a.shape[1]]))
        # logSkewZ.append()
        
    return maxT, betaList, np.array(meanF), np.array(varF), np.array(skewF), np.array(logMeanZ), np.array(logVarZ) #np.array(meanLog), np.array(varLog), np.array(skewLog)
# Scaling of mean: (mean/(tMax-2)/np.log(4) - 1 ) / beta
# Scaling of var: var / beta**2

def plotMeanF(maxT, betaList, meanF, dim=2):
    times = np.unique(maxT)
    for t in times:
        print(t)
        indexArr = (maxT == t)
        plt.loglog(betaList[indexArr], -betaList[indexArr] * (meanF[indexArr]/np.log(2**dim)/t) -1 , '-o', label=f'tMax = {t}', mfc='none')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$-\beta \langle F\rangle /(N \ln(2^d)) - 1$')
    # plt.ylim([.5,20])
    plt.legend()
    plt.show()

def plotMeanEntropy(maxT, betaList, meanF, dim = 2):
    tempList = 1/betaList
    times = np.unique(maxT)
    for t in times:
        print(t)
        indexArr = (maxT == t)
        temp = tempList[indexArr]
        scaledF = meanF[indexArr]/(dim*np.log(2))/t
        deltaTemp = np.diff(temp)
        entropy = -np.diff(scaledF)/deltaTemp
        plt.semilogx(1/(temp[:-1] + deltaTemp/2), entropy, '-o', label=f'tMax = {t}', mfc='none')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$S$')
    # plt.ylim([.5,20])
    plt.legend()
    plt.show()

def plotVarF(maxT, betaList, varF, dim, linewidth=1):
    times = np.unique(maxT)
    variancePrediction = np.array([computeVariancePrediction(N, dim) for N in times])
    print(variancePrediction)
    for i, t in enumerate(times):
        indexArr = (maxT == t)
        # plt.semilogx(betaList[indexArr], (varF[indexArr]/variancePrediction[i]), '-o', linewidth=linewidth, label=f'tMax = {t}', mfc='none')
        # plt.loglog(betaList[indexArr], (varF[indexArr]/variancePrediction[i]) - 1, '-o', linewidth=linewidth, label=f'tMax = {t}', mfc='none')
        plt.semilogx(betaList[indexArr], (varF[indexArr] - variancePrediction[i])/(varF[indexArr] - variancePrediction[i])[-1], '-o', label=f'tMax = {t}', mfc='none')
        # plt.semilogx(betaList[indexArr], (varF[indexArr] - variancePrediction[i]/(1 + betaList[indexArr]**2/4))/(varF[indexArr] - variancePrediction[i]/(1 + betaList[indexArr]**2/4))[-1], '-o', label=f'tMax = {t}', mfc='none')
    plt.gca().set_prop_cycle(None)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$Var(F)/$(small $\beta$ prediction from text)')
    plt.legend()
    plt.show()

def plotSkewF(maxT, betaList, skewF, linewidth=1):
    times = np.unique(maxT)
    for t in times:
        indexArr = (maxT == t)
        plt.semilogx(betaList[indexArr], skewF[indexArr], '-o', linewidth=linewidth, label=f'tMax = {t}', mfc='none')
    plt.gca().set_prop_cycle(None)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$Skew(F)$')
    plt.legend()
    plt.show()

import time
def singleEvolution(tMax, beta0, beta, dim=2, new=True):    
    pointToPlane = []
    elapsedTime = []
    if new or dim >2:
        it = transferMatrixND(dim, tMax, beta0*beta)
    else:
        it = transferMatrix2D(tMax, beta0*beta)
    for logZ, t in it:
        if t == 0:
            s = time.time()
        # pointToPlane.append(logSumExp(logZ.reshape(tMax, tMax)[:t+1,:t+1].flatten()))
        pass
        # pointToPlane.append(logSumExp(logZ))
        elapsedTime.append(time.time()-s)
        print(t, elapsedTime[-1])
    return logZ, np.array(elapsedTime)

def varianceCheck(nSystems, N, beta0):
    allLogZ = []
    for sys in range(nSystems):
        allLogZ.append(singleEvolution(N, beta0, np.ones(N)))
        if np.mod(sys, 100) == 0:
            print(sys)
    
    F = -np.array(allLogZ)/beta0
    return F

def computeVariance(sys, N, beta0):
    F = -singleEvolution(N, beta0, np.ones(N))/beta0
    return F

def computeVariancePrediction(N, dim):
    secondMoment = 0
    for n in range(N+1):
        sumTerm = (scipy.special.binom(2*n, n) * 2**(-2*n))**dim
        if not np.isfinite(sumTerm):
            # sumTerm = 1/(n*np.pi)
            sumTerm = (np.pi * n)**(-dim/2)
        secondMoment += sumTerm
    return secondMoment

# def processLogZFiles(globString):
#     # Read through all of the data files and sort each entry into the appropriate file
#     # Each file should contain the information for a given (time, beta0) pair
#     # If there are 5 times and 9 beta0s then there should be 45 output files
#     # This can be done with awk, using
#     # awk -F", " '{file = "t=" $1 ",beta0=" sprintf("%.14f",$2) ".dat"; print $3 FS $4 FS $5 >> file}' *.dat
    
#     all = []
#     for f in glob.glob(globString):
        
#         data = readLogZFiles(globString, tList, beta0List, nMeasurements=3)
#     # We want to reshape the data into a dictionary
#     # [time, beta0, measurementId, element]
#     # if tMax=1000, there will be 5 times, 9 beta0s, 3 measurementIds, N elements

if __name__ == "__main__":

    # Call as `python3 directedPolymer.py dim tMax numSystems outFile betaString`
    inputIndex = 1
    dim = int(sys.argv[inputIndex]); inputIndex += 1
    tMax = int(sys.argv[inputIndex]); inputIndex += 1
    numSystems = int(sys.argv[inputIndex]); inputIndex += 1
    outFileName = sys.argv[inputIndex]; inputIndex += 1
    betaString = sys.argv[inputIndex]; inputIndex += 1 # Example string, "np.ones(tMax)", 
    
    measurementTimes = np.geomspace(1,tMax, np.round(2 * np.log10(tMax)).astype(int) + 1).astype(int)
    # make tMax one more so that our measurements end at the input value, rather than input value-1
    # print(measurementTimes)
    tMax += 1
    beta = eval(betaString)
    # beta0List = np.geomspace(.1,10,9)
    beta0List = np.geomspace(.01,100,17)
    # beta0List = np.geomspace(.1*10**(33/64),10**(31/64), 10)
    # s = time.time()
    with open(outFileName, 'a') as file:        
        for _ in range(numSystems):
            for beta0 in beta0List:
                measureIndex = 0
                for logZ, t in transferMatrixND(dim, tMax, beta0*beta):
                    # Make measurements that are log-spaced
                    if measurementTimes[measureIndex] == t:
                        measureIndex += 1
                        measurements = measurePartitionFunction(logZ, tMax, dim)
                        # measurementTimes = np.delete(measurementTimes, 0)
                        file.write(f"{t}, {beta0}, {', '.join(map(str, measurements))} \n")
                        # print(t, time.time()-s, p2Plane, p2Line, p2Point)
                file.flush()


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
