import numpy as np
import os
from time import time as wallTime
from numba import njit
import json
# import tracemalloc
from datetime import date
import sys
import shutil
import math
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from directedPolymer import logSumExp
from scipy.optimize import curve_fit

# This module is intended to evolve the PMF of the difference random walk
# defined as \vec{V(t)} = \vec{R1(t)} - \vec{R2(t)}
# under the tilted two-point measure defined in Jacob's 2d Random Walk overleaf
# document, specifically eqns 19 and 20
# the goal is ultimately to be able to compute kappa, and thus the invariant
# measure mu, to get a scaling for the local time.

# don't worry about numba or logsumExp or storing things as logP yet
# assumes dirichlet,

def makeDirectionList(nDim=2):
    return np.vstack([np.eye(nDim, dtype=int), -np.eye(nDim,dtype=int)])

def calcXiExpectation(n1, n2, alpha, correlated=False):
    """ calculates E_nu[xi(n1)xi(n2)] assumming xi's are Dirichlet distributed with parameter alpha"""
    alpha0 = 4*alpha
    if correlated:
        if (n1 == n2).all():
            # This should actually be alpha_i * (alpha_i + 1) / (alpha0 * (alpha0 + 1))
            return alpha * (alpha + 1) / (alpha0 * (alpha0 + 1))
        else:
            # This should actually be alpha_i * alpha_j / (alpha0 * (alpha0 + 1))
            return alpha**2 / (alpha0 * (alpha0 + 1))
    else:
        return alpha**2 / alpha0**2

def twoWalkerTransitionProbabilities(alpha, v, tiltDirection = np.array([1,0]), correlated=False):
    # These are the transition probabilites for a pair of walkers. Since each walker can only move one step
    # in the cardinal directions we will return
    # list of directions1, list of directions2, transition probability

    # First, generate the list of directions
    directions = makeDirectionList(nDim=2)

    #compute the denominator
    denominator = 0
    for m1 in directions:
        for m2 in directions:
            denominator += calcXiExpectation(m1, m2, alpha, correlated=correlated) * np.exp( 2 * np.arctanh(v) * np.dot( (m1 + m2), tiltDirection))

    d1List = []
    d2List = []
    probabilityList = []
    for n1 in directions:
        for n2 in directions:
            d1List.append(n1)
            d2List.append(n2)

            prefactor = calcXiExpectation(n1, n2, alpha, correlated=correlated)
            expTerm = np.exp( 2 * np.arctanh(v) * np.dot( (n1 + n2), tiltDirection))
            probabilityList.append(prefactor * expTerm / denominator)
            # print(n1, n2, prefactor, expTerm, denominator)

    return np.array(d1List), np.array(d2List), np.array(probabilityList)

def computeVTransitionProbabilities(alpha, v, tiltDirection = np.array([1,0]), correlated=False, maxStep=2):
    # Return the transition matrix that tells us the probability of moving from a site V to V+[i,j]
    width = maxStep*2 + 1
    transition = np.zeros((width, width))

    d1, d2, p = twoWalkerTransitionProbabilities(alpha, v, tiltDirection=tiltDirection, correlated=correlated)    
    deltaV = d1 - d2
    for i in range(len(p)):
        # print(deltaV[i], p[i])
        transition[deltaV[i,0] + maxStep, deltaV[i,1] + maxStep] += p[i]
    return transition

# def evolvePofVConvolve(tMax, alpha, v, maxStep=2):
#     pmfs = []
#     # atOrigin = calcTransitionMatrix(alpha, v, origin=True)
#     atOrigin = computeVTransitionProbabilities(alpha, v, correlated=True)
#     # notAtOrigin = calcTransitionMatrix(alpha, v, origin=False)
#     notAtOrigin = computeVTransitionProbabilities(alpha, v, correlated=False)
#     # Initial condition is two walks at the same site, so PMF=1 at the origin
#     PofV = np.array([[1]])
#     pmfs.append(PofV.copy())
#     for t in range(tMax):
#         # TODO: wrong next line
#         originValue = PofV[t,t]
#         # We can use scipys convolve to get the results everywhere but at the origin
#         PofV = convolve2d(PofV, notAtOrigin, mode='full', boundary='fill', fillvalue=0)
#         # After the convolution we will have a new matrix that is 2 elements larger in each dimension
#         # So, the location of the new origin will be at 2t+1, 2t+1
#         # And then we have to subtract off the wrong convolution and add in the right one at the origin
#         # TODO: wrong next line
#         newOrigin=2*t + 2 # At t=0 this value needs to be 2, at t=1 it needs to be 4, etc.
#         PofV[newOrigin-maxStep:newOrigin+maxStep+1, newOrigin-maxStep:newOrigin+maxStep+1] += originValue * (atOrigin - notAtOrigin)
#         pmfs.append(PofV.copy())
#     return pmfs

def evolvePofVGold(tMax, alpha, v, maxStep=2):
    pmfs = []
    # atOrigin = calcTransitionMatrix(alpha, v, origin=True)
    atOrigin = computeVTransitionProbabilities(alpha, v, correlated=True)
    # notAtOrigin = calcTransitionMatrix(alpha, v, origin=False)
    notAtOrigin = computeVTransitionProbabilities(alpha, v, correlated=False)
    # Initial condition is two walks at the same site, so PMF=1 at the origin
    PofV = np.array([[1]])
    pmfs.append(PofV.copy())
    # We could get an 8-fold speedup by exploiting the 8-fold symmetry of the plane and using reflecting boundary conditions
    for t in range(tMax):
        newPofV = np.zeros(np.array(PofV.shape) + 2*maxStep)
        for i in range(PofV.shape[0]):
            for j in range(PofV.shape[1]):
                # By hand apply the transition matrices
                if i == 2*t and j == 2*t:
                    newPofV[i:i+2*maxStep+1, j:j+2*maxStep+1] += PofV[i,j]*atOrigin
                else:
                    newPofV[i:i+2*maxStep+1, j:j+2*maxStep+1] += PofV[i,j]*notAtOrigin
        # We can use scipys convolve to get the results everywhere but at the origin
        PofV = newPofV
        pmfs.append(PofV.copy())
    return pmfs

def evolvePofVLSEGold(tMax, alpha, v, maxStep=2):
    logpmfs = []
    # atOrigin = calcTransitionMatrix(alpha, v, origin=True)
    atOrigin = np.log(computeVTransitionProbabilities(alpha, v, correlated=True))
    # notAtOrigin = calcTransitionMatrix(alpha, v, origin=False)
    notAtOrigin = np.log(computeVTransitionProbabilities(alpha, v, correlated=False))
    # Initial condition is two walks at the same site, so PMF=1 at the origin and logPMF=0
    PofV = np.array([[np.log(1)]])
    logpmfs.append(PofV.copy())
    # We could get an 8-fold speedup by exploiting the 8-fold symmetry of the plane and using reflecting boundary conditions
    for t in range(tMax):
        newPofV = np.full(np.array(PofV.shape) + 2*maxStep, -np.inf)
        for i in range(PofV.shape[0]):
            for j in range(PofV.shape[1]):
                # By hand apply the transition matrices
                for m in range(2*maxStep+1):
                    for n in range(2*maxStep+1):
                        transferValue = atOrigin[m, n] if (i == 2*t and j == 2*t) else notAtOrigin[m,n]
                        newPofV[i+m, j+n] = logSumExp(np.array([newPofV[i+m, j+n], PofV[i,j] + transferValue ]))
        # We can use scipys convolve to get the results everywhere but at the origin
        PofV = newPofV
        logpmfs.append(PofV.copy())
    return logpmfs

@njit
def calcNextPMF(t, logPofV, atOrigin, notAtOrigin, maxStep=2):
    newSize = logPofV.shape[0] + 2*maxStep
    newPofV = np.full((newSize, newSize), -np.inf) # Fill with np.log(0)
    # newPofV[:] = -np.inf
    # newPofV = np.zeros(np.array(PofV.shape) + 2*maxStep).fill(-np.inf)
    # TODO: There's another 8x speedup by only looking at half of a quadrant
    for i in range(logPofV.shape[0]):
        for j in range(logPofV.shape[1]):
            # By hand apply the transition matrices, but only if the element is not equal to -np.inf
            if np.isfinite(logPofV[i,j]):
                for m in range(2*maxStep+1):
                    for n in range(2*maxStep+1):
                        transferValue = atOrigin[m, n] if (i == maxStep*t and j == maxStep*t) else notAtOrigin[m,n]
                        if np.isfinite(transferValue):
                            newPofV[i+m, j+n] = logSumExp(np.array([newPofV[i+m, j+n], logPofV[i,j] + transferValue ]))
    # We can use scipys convolve to get the results everywhere but at the origin
    return newPofV

def evolvePofVNumba(tMax, alpha, v, maxStep=2):
    atOrigin = np.log(computeVTransitionProbabilities(alpha, v, correlated=True))
    notAtOrigin = np.log(computeVTransitionProbabilities(alpha, v, correlated=False))
    # Initial condition is two walks at the same site, so PMF=1 at the origin and logPMF=0
    logPofV = np.array([[np.log(1)]])
    logpmfs = [logPofV.copy()]
    s = wallTime()
    for t in range(tMax):
        logPofV = calcNextPMF(t, logPofV, atOrigin, notAtOrigin, maxStep=maxStep)
        logpmfs.append(logPofV.copy())
        print(t, wallTime()-s)
    return logpmfs

def computeDegeneracy(size, checkerboard=False):
    x, y = np.meshgrid(range(-size,size+1), range(-size, size+1))
    x = x.flatten()
    y = y.flatten()
    dsq = (x**2 + y**2)
    # NOTE: This is a slightly dangerous thing to do as it assumes that we're using a checkerboard
    if checkerboard:
        # We only want the entries for which x+y is even
        good = (np.mod(x + y,2) == 0)
        dsq = dsq[good]
        x = x[good]
        y = y[good]
    # Sort dsq by distance
    s = np.argsort(dsq)
    dsq = dsq[s]
    x = x[s]
    y = y[s]
    # Find the location of the unique values, which can be used for summing
    dsqVal, index = np.unique(dsq, return_index=True)
    # We can only trust the degeneracy for distances up to size
    good = (dsqVal <= size**2)
    dsqVal = dsqVal[good]
    # We need to keep one more element of index so that we can actually finish the counting
    good[np.sum(good)] = True
    index = index[good]

    degen = np.diff(index)
    siteList = []
    for i in range(len(dsqVal)):
        elements = slice(index[i], index[i+1])
        siteList.append(np.array([x[elements], y[elements]]).T)

    return dsqVal, index, degen, s, siteList

def computeInvariantMeasure(PMF):
    # compute the distance squared to each point, the indexing, the degeneracy, and the sorting
    dsqVal, index, degen, s, _ = computeDegeneracy(PMF.shape[0]//2, checkerboard=False)
    sortPMF = PMF.flatten()[s]
    logInvariantMeasure = np.array([logSumExp(sortPMF[index[i]:index[i+1]]) for i in range(dsqVal.shape[0])])
    # Normalize invariant measure so that the 0th element is 1
    logInvariantMeasure -= logInvariantMeasure[0]
    # Remove the points that are -np.inf
    good = np.isfinite(logInvariantMeasure)
    return np.sqrt(dsqVal[good]), logInvariantMeasure[good], degen[good]

def analyticInvariantMeasure(alpha, v, dMax, maxStep=2, numeric=False):
    if numeric:
        # First, compute the transition matrices
        atOrigin = computeVTransitionProbabilities(alpha, v, correlated=True)
        notAtOrigin = computeVTransitionProbabilities(alpha, v, correlated=False)
        # Now compute the ratio between the probability at the origin and the probability at any other site
        mu = (1 - atOrigin[maxStep, maxStep]) / (1 - notAtOrigin[maxStep, maxStep])
    else:
        mu = (4 * alpha) / (4 * alpha + (1 + v**2)**2)
    # Now compute the degeneracy at allowed lattice sites.  Recall that an allowed lattice point is one for which i+j is even since we're on the checkerboard
    # compute the distance squared to each point, the indexing, the degeneracy, and the sorting
    dsqVal, index, degen, s, _ = computeDegeneracy(int(dMax), checkerboard=True)
    # We only want to include distances that are less than or equal to dMax
    logInvariantMeasure = np.log(mu) + np.log(degen)
    logInvariantMeasure[0] = 0

    return np.sqrt(dsqVal), logInvariantMeasure, mu

# the below eqns are to calculate kappa(l).

# eqn 53 (or eqn 54), kappa for ||l|| != 0
def computeLVecExpectation(lVec, v, alpha,correlated=False):
    """ calculates eqn 53, given a vector l. (over the combinations of n1 and n2)
    lVec: 2d vector array
    v: float between 0 and 1
    alpha: float between 0 and infinity
    """
    # return the probabilities for the 16 combos of n1 and n2. always uncorrelated for this
    # note the twoWalkerTransitionProbabilities implicity has the (1-v^2)^2 / 16 term in it
    n1s, n2s, probs = twoWalkerTransitionProbabilities(alpha, v, tiltDirection = np.array([1,0]), correlated=correlated)
    sumOverSites = 0
    for i in range(len(n1s)):
        logTerm = np.log((1 + np.linalg.norm(lVec + n1s[i] - n2s[i])) / (1 + np.linalg.norm(lVec)))
        sumOverSites += logTerm * probs[i]
    return sumOverSites


# the following function combines the above 2 functions but uses computeDegeneracies instead of find_two_squares
def computeAllKappa(alpha, v, dMax, checkerboard=True):
    """ compute kappa(l) up to some maximum l^2 value (equiv, max distance l), given v (measurement velocity)
    and alpha (stickiness of Dirichlet distribution).
    The checkerboard flag is if we want to be smart about calculating it"""
    dMax = int(dMax)
    # value of lSq, its degenerecy, and the list of sites that have that distance
    dsqVals, _, degeneracies, _, siteLists = computeDegeneracy(dMax, checkerboard=checkerboard)
    kappaList = np.zeros_like(dsqVals,dtype=float)
    for idx, dSqVal in enumerate(dsqVals):
        kappa = 0
        correlated = True if dSqVal == 0 else False
        for site in siteLists[idx]:
            kappa += computeLVecExpectation(site, v, alpha,correlated=correlated)
        kappa /= degeneracies[idx]
        kappaList[idx] = kappa
    return np.sqrt(dsqVals), kappaList

@njit
def computeKappaTerm(latticeVector, n1s, n2s, probs):
    kappaTerm = 0
    vNorm = np.sqrt(np.sum(latticeVector**2))
    for n1, n2, prob in zip(n1s, n2s, probs):
        vPrimeNorm = np.sqrt(np.sum((latticeVector + n1 - n2)**2))
        kappaTerm += np.log((1 + vPrimeNorm) / (1 + vNorm)) * prob
        # kappaTerm += np.log((1 + np.linalg.norm(latticeVector + n1 - n2)) / (1 + np.linalg.norm(latticeVector))) * prob
    return kappaTerm

def computeKappaMuProduct(alpha, v, dMax):
    # s = wallTime()
    muRatio = (4 * alpha) / (4 * alpha + (1 + v**2)**2)

    dSqVals, _, _, _, siteList = computeDegeneracy(dMax, checkerboard=True)
    # print(f'computeDegeneracy={wallTime()-s}')
    # s = wallTime()
    kappaMuProductList = np.zeros_like(dSqVals,dtype=float)
    n1s, n2s, probsCorrelated = twoWalkerTransitionProbabilities(alpha, v, correlated=True)
    _,   _,   probsUncorrelated = twoWalkerTransitionProbabilities(alpha, v, correlated=False)
    for i, dSqVal in enumerate(dSqVals):
        for site in siteList[i]:
            if dSqVal == 0:                
                kappaMuProductList[i] += computeKappaTerm(site, n1s, n2s, probsCorrelated) * 1
            else:
                kappaMuProductList[i] += computeKappaTerm(site, n1s, n2s, probsUncorrelated) * muRatio
    # print(f'computeLVecExpectation={wallTime()-s}')
    return np.sqrt(dSqVals), kappaMuProductList

def computeG(alpha, v):
    numerator = 4 * (4 * alpha  + (v**2 + 1)**2)
    denominator = (4 * alpha + 1) * (v**2 + 1)**2
    return np.log(numerator / denominator)

def fitBetaSq(kappaMu, g):
    def powerLaw(x, a, alpha, b):
        return a * x**alpha + b
    x = np.arange(1, len(kappaMu)+1)
    cumSum = np.cumsum(kappaMu)
    # Scale the starting point to 1
    cumSum = cumSum[-1] - cumSum
    cumSum /= cumSum[0]
    popt, pcov = curve_fit(powerLaw, x, cumSum, p0=[1, -.55, 0])
    # The "True" value of the kappaMu sum needs to be increased by a factor of 1 - b, where b is the offset
    trueKappaMuSum = np.sum(kappaMu) - popt[2]*kappaMu[0]

    return g/trueKappaMuSum#, g / np.sum(kappaMu)

def computeBeta(alpha, v, dMax=100):
    _, kappaMuProduct = computeKappaMuProduct(alpha, v, dMax)
    g = computeG(alpha,v)
    betaSq = fitBetaSq(kappaMuProduct, g)
    return np.sqrt(betaSq)

def finiteImshow(im):
    a = im.copy()
    alpha = np.isfinite(a)
    a[~alpha] = np.min(alpha)
    plt.imshow(a, alpha=alpha.astype(float))

def computeMoment(logPMF, moment):
    size = logPMF.shape[0]//2
    x,y = np.meshgrid(range(-size, size+1), range(-size, size+1))
    dsq = x**2 + y**2
    del x, y
    return np.sum(dsq**(moment/2)*np.exp(logPMF))

def computeVarMagV(logPMF):
    return computeMoment(logPMF,2) - computeMoment(logPMF, 1)**2

def naiveSingleParticleTransition(v, tiltDirection = np.array([1,0])):
    directions = makeDirectionList(nDim=2)
    probability = []
    for n in directions:
        probability.append((1-v**2)/4 * np.exp(2 * np.arctanh(v) * np.dot(n, tiltDirection)))
    pOut = np.zeros((3,3))
    for i in range(len(probability)):
        pOut[directions[i,0]+1, directions[i,1]+1] += probability[i]
    return pOut

def makeSingleParticleTransition(d1,d2, p):
    p1 = np.zeros((3,3))
    p2 = np.zeros((3,3))
    for i in range(16):
        p1[d1[i,0]+1, d1[i,1]+1] += p[i]
        p2[d2[i,0]+1, d2[i,1]+1] += p[i]
    return p1, p2

def find_two_squares(targetDistSq):
    """ Finds all sites on lattice that have a norm-squared of targetDistSq, i.e.
    find all \vec{x} such that ||x||^2 = targetDistSq
    inputs:
        targetDistSq: integer (because distSq of points on a lattice will always be integers)"""
    pairs = []
    # Left pointer starts at 0, right pointer starts at the square root of target
    left = 0
    right = int(np.sqrt(targetDistSq))
    while left <= right:
        current_sum = left ** 2 + right ** 2
        if current_sum == targetDistSq:
            # deal with octant symmetry?
            if left == 0 and right == 0:  # origin
                # print("origin")
                pairs.append((left, right))
            elif left == right: # along a diagonal
                # print("diagonal")
                pairs.append((left, right))
                pairs.append((left, -right))
                pairs.append((-left, right))
                pairs.append((-left, -right))
            elif left == 0:  # cardinal
                # print("cardinal")
                pairs.append((right, 0))
                pairs.append((-right, 0))
                pairs.append((0, right))
                pairs.append((0, -right))
            else:  # off-diagonal and off-0
                # print("general")
                pairs.append((left, right))
                pairs.append((left, -right))
                pairs.append((-left, -right))
                pairs.append((-left, right))
                pairs.append((right, left))
                pairs.append((-right, left))
                pairs.append((-right, -left))
                pairs.append((right, -left))
            left += 1
            right -= 1
        elif current_sum < targetDistSq:
            left += 1
        else:
            right -= 1
    return pairs

# def computeKappaOfLNonzero(lSq, v, alpha):
#     """ compute kappa(l) given l^2. to do this we need to find all sites with distance^2 = l^2 and then
#     divide by the number of sites
#
#     currently this works by doing one distance^2 at a time."""
#     kappa = 0
#     sites_with_distSq_L = find_two_squares(lSq)
#     if len(sites_with_distSq_L) == 0:
#         return np.nan
#     else:
#         for site in sites_with_distSq_L:
#             kappa += computeLVecExpectation(site, v, alpha)
#         kappa /= len(sites_with_distSq_L)
#         return kappa
#
#
# def getKappaList(lSqMax, v, alpha):
#     # initialize with the l=0 case, of which there's only one site, [0,0]. This is the only case where we need to set correlated=True
#     lSqList = [0]
#     kappaList = [computeLVecExpectation(np.array([0,0]), v, alpha,correlated=True)]
#     for i in range(1, lSqMax + 1):
#         lSqList.append(i)
#         kappaList.append(computeKappaOfLNonzero(i, v, alpha))
    return np.array(lSqList), np.array(kappaList)



# ------------------------------------------- Everything below here may be based on wrong assumptions


# def normalization(r1, alpha, v):
#     """ returns alpha^2 (eqn. 23) if r1 = 0 or the 1-v shit if not"""
#     if (r1 == [0,0]).all():
#         nHats = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
#         xHat = nHats[0]
#         # sum n1 + n2 = r2 - r1 for n1 and n2 in set of nhats
#         normalization = 0
#         for n1 in nHats:
#             for n2 in nHats:
#                 prefactor = calcXiExpectation(n1, n2, alpha)
#                 normalization += prefactor * np.exp(2 * np.arctanh(v) * np.dot(xHat, (n1 + n2)))
#         return normalization
#     else:  # note: this is inverted because when we call this function we use /=
#         return 16 / ((1 - v**2)**2 )
#
#
# def calcTransitionProb(r1, r2, alpha, v):
#     """ this should return a proability for the change from r1 to r2; implemented eqn. 22"""
#     r1 = np.asarray(r1) if isinstance(r1, list) else r1
#     r2 = np.asarray(r2) if isinstance(r2, list) else r2
#     diff = r2 - r1
#     nHats = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
#     xHat = nHats[0]
#     # sum n1 + n2 = r2 - r1 for n1 and n2 in set of nhats
#     prob = 0
#     for n1 in nHats:
#         for n2 in nHats:
#             if ((n1 + n2) == diff).all():
#                 # print(f"Allowed values {n1, n2}")
#                 # For now, only consider the case where r1 == [0,0]
#                 if (r1 == [0,0]).all():  # if the 2 walks are at the same site
#                     prefactor = calcXiExpectation(n1, n2, alpha)
#                 else:  # if the 2 walks are separated
#                     prefactor = 1
#                 prob += prefactor * np.exp(2 * np.arctanh(v) * np.dot(xHat, (n1 + n2)))
#     prob /= normalization(r1, alpha, v)  # this gives alpha^2
#     return prob
#
# def calcTransitionMatrix(alpha, v, origin=True, maxStep=2):
#     width = maxStep*2 + 1
#     transition = np.zeros((width, width))
#     if origin:
#         r1 = np.array([0,0])
#     else:
#         r1 = np.array([1,1])
#     for i in range(width):
#         for j in range(width):
#             transition[i,j] = calcTransitionProb(r1, r1 + np.array([i-maxStep,j-maxStep]), alpha, v)
#     return transition

# The following commented out functions are old and not quite right.
# def calcDelta(V):
#     """ calculates delta(t) = ln(1 + ||V(t)||)"""
#     return np.log(1 + np.linalg.norm(V))
#
#
# def calcDeltaDifference(V1, V2):
#     """ calculates the difference between two deltas"""
#     return np.log((1 + np.linalg.norm(V2)) / (1 + np.linalg.norm(V1)))
#
#
# def calcKappaVVec(V, alpha, v, maxStep=2):
#     """ returns kappa for a speciic value of vec V = vec r"""
#     # Create the possible neighboring sites of vec V
#     x, y = np.meshgrid(np.arange(-maxStep, maxStep+1), np.arange(-maxStep, maxStep+1))
#     pos = V + np.vstack([x.flatten(), y.flatten()]).T
#     # develop
#     kappa = 0
#     # do the expectation over the allowed values of the delta difference
#     for site in pos:
#         kappa += calcDeltaDifference(V, site) * calcTransitionProb(V, site, alpha, v)
#     return kappa
#
#
# def createKappaGrid(alpha, v, size=10):
#     """
#     cretes a grid where the x and y directions correspond to the x and y components of r
#     then the value at each site is the value of kappa(\vec{r})
#     note: this will be slow if size is large
#     """
#     # get values of x and y
#     x, y = np.meshgrid(np.arange(-size,size+1), np.arange(-size, size+1))
#     # turn them into sets of coordinates which populate the grid
#     rArraySites = np.vstack([x.flatten(), y.flatten()]).T
#     # initialize
#     kappaArray = np.zeros_like(x,dtype=float)  #x is integers so we need to cast it to float
#     for site in rArraySites:
#         kappaArray[site[0]+size,site[1]+size] = calcKappaVVec(site, alpha, v)
#     return rArraySites, kappaArray
#
#
# # def kappaOfR(rArraySites, kappaArray):
# #     """ turn grid of kappa(vec r) into kappa(r) """
# #     data = []
# #     for site in rArraySites:
# #         data.append([np.linalg.norm(site), kappaArray[site[0],site[1]]])
# #     data = np.array(data)
# #     avgs = []
# #     # now go through and find places where all rs are equal and calc the avg kappa
# #     for r in data[:,0]:
# #         avgs.append(np.mean(data[data[:,0] == r,1]))
# #     return data[:,0], np.array(avgs)
#
# def kappaOfRFixed(rArraySites, kappaArray,size):
#     """ trn grid of kappa(vec r) into kappa(r)"""
#     data = []
#     for site in rArraySites:
#         data.append([np.linalg.norm(site), kappaArray[site[0]+size,site[1]+size]])
#     data = np.array(data)
#     avgs = []
#     uniqueR = np.unique(data[:,0])
#     for r in uniqueR:
#         indices = (data[:,0] == r)
#         avgs.append(np.mean(data[indices,1]))
#     return data, uniqueR, avgs
#
# # the following functions are to maybe deal with the weirdness at the corners
# # by creating a grid and then averaging over all sites with distance r, we're actually
# # missing some spots because a square lattice won't have every site that has distance r for the
# # corners of the square.
# def find_two_squares(target):
#     """ this algorithm finds all coordinates in an octant that would fall on a circle"""
#     pairs = []
#     # left pointer starts at 0, right pointer starts at the sqrt of the target
#     left = 0
#     right = int(math.sqrt(target))
#     while left <= right:
#         current_sum = left**2 + right ** 2
#         if current_sum == target:
#             pairs.append(left, right)
#             left += 1
#             right -= 1
#         elif current_sum < target:
#             left += 1
#         else:
#             right -= 1
#     return pairs

def comparePMFs(pmf1, alpha1, pmf2, alpha2, fileName='temp/kappa/'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    for i in range(len(pmf1)):
        ax1.imshow(pmf1[i])        
        ax1.set_axis_off()
        ax1.set_title(f'alpha={alpha1}')
        ax2.imshow(pmf2[i])
        ax2.set_axis_off()
        ax2.set_title(f'alpha={alpha2}')
        # plt.tight_layout()
        plt.savefig(f'{fileName}{i:05d}.png', bbox_inches='tight')

def comparePMFsJoined(pmf1, alpha1, pmf2, alpha2, fileName='temp/kappa/'):
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

    for i in range(len(pmf1)):
        plt.imshow(np.hstack([pmf1[i],pmf2[i]]))
        # plt.tight_layout()
        plt.savefig(f'{fileName}{i:05d}.png', bbox_inches='tight')
