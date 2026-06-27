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


# This module is intended to evolve the PMF of the difference random walk
# defined as \vec{V(t)} = \vec{R1(t)} - \vec{R2(t)}
# under the tilted two-point measure defined in Jacob's 2d Random Walk overleaf
# document, specifically eqns 19 and 20
# the goal is ultimately to be able to compute kappa, and thus the invariant
# measure mu, to get a scaling for the local time.

# don't worry about numba or logsumExp or storing things as logP yet
# assumes dirichlet,


def normalization(r1, alpha, v):
    """ returns alpha^2 (eqn. 23) if r1 = 0 or the 1-v shit if not"""
    if (r1 == [0,0]).all():
        nHats = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
        xHat = nHats[0]
        # sum n1 + n2 = r2 - r1 for n1 and n2 in set of nhats
        normalization = 0
        for n1 in nHats:
            for n2 in nHats:
                prefactor = calcXiExpectation(n1, n2, alpha)
                normalization += prefactor * np.exp(2 * np.arctanh(v) * np.dot(xHat, (n1 + n2)))
        return normalization
    else:  # note: this is inverted because when we call this function we use /=
        return 16 / ((1 - v**2)**2 )


def calcXiExpectation(n1, n2, alpha):
    """ calculates E_nu[xi(n1)xi(n2)] assumming xi's are Dirichlet distributed with parameter alpha"""
    if (n1 == n2).all():
        return alpha * (alpha + 1) / (4 * alpha * (4 * alpha + 1))
    else:
        return alpha ** 2 / (4 * alpha * (4 * alpha + 1))


def calcTransitionProb(r1, r2, alpha, v):
    """ this should return a proability for the change from r1 to r2; implemented eqn. 22"""
    r1 = np.asarray(r1) if isinstance(r1, list) else r1
    r2 = np.asarray(r2) if isinstance(r2, list) else r2
    diff = r2 - r1
    nHats = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
    xHat = nHats[0]
    # sum n1 + n2 = r2 - r1 for n1 and n2 in set of nhats
    prob = 0
    for n1 in nHats:
        for n2 in nHats:
            if ((n1 + n2) == diff).all():
                # print(f"Allowed values {n1, n2}")
                # For now, only consider the case where r1 == [0,0]
                if (r1 == [0,0]).all():  # if the 2 walks are at the same site
                    prefactor = calcXiExpectation(n1, n2, alpha)
                else:  # if the 2 walks are separated
                    prefactor = 1
                prob += prefactor * np.exp(2 * np.arctanh(v) * np.dot(xHat, (n1 + n2)))
    prob /= normalization(r1, alpha, v)  # this gives alpha^2
    return prob

def calcTransitionMatrix(alpha, v, origin=True):
    transition = np.zeros((3,3))
    if origin:
        r1 = np.array([0,0])
    else:
        r1 = np.array([1,1])
    for i in range(3):
        for j in range(3):
            transition[i,j] = calcTransitionProb(r1, r1 + np.array([i-1,j-1]), alpha, v)
    return transition
            

def evolvePofV(tMax, alpha, v):
    pmfs = []
    atOrigin = calcTransitionMatrix(alpha, v, origin=True)
    notAtOrigin = calcTransitionMatrix(alpha, v, origin=False)
    # Initial condition is two walks at the same site, so PMF=1 at the origin
    PofV = np.array([[1]])
    pmfs.append(PofV.copy())
    for t in range(tMax):
        originValue = PofV[t,t]
        # We can use scipys convolve to get the results everywhere but at the origin
        PofV = convolve2d(PofV, notAtOrigin, mode='full', boundary='fill', fillvalue=0)
        # After the convolution we will have a new matrix that is 2 elements larger in each dimension
        # So, the location of the new origin will be at t+1, t+1
        # And then we have to subtract off the wrong convolution and add in the right one at the origin
        PofV[t:t+3,t:t+3] += originValue * (atOrigin - notAtOrigin)
        pmfs.append(PofV.copy())
    return pmfs

def calcDelta(V):
    """ calculates delta(t) = ln(1 + ||V(t)||)"""
    return np.log(1 + np.linalg.norm(V))


def calcDeltaDifference(V1, V2):
    """ calculates the difference between two deltas"""
    return np.log((1 + np.linalg.norm(V2)) / (1 + np.linalg.norm(V1)))


def calcKappaVVec(V, alpha, v, maxStep=2):
    """ returns kappa for a speciic value of vec V = vec r"""
    # Create the possible neighboring sites of vec V
    x, y = np.meshgrid(np.arange(-maxStep, maxStep+1), np.arange(-maxStep, maxStep+1))
    pos = V + np.vstack([x.flatten(), y.flatten()]).T
    # develop
    kappa = 0
    # do the expectation over the allowed values of the delta difference
    for site in pos:
        kappa += calcDeltaDifference(V, site) * calcTransitionProb(V, site, alpha, v)
    return kappa


def createKappaGrid(alpha, v, size=10):
    """
    cretes a grid where the x and y directions correspond to the x and y components of r
    then the value at each site is the value of kappa(\vec{r})
    note: this will be slow if size is large
    """
    # get values of x and y
    x, y = np.meshgrid(np.arange(-size,size+1), np.arange(-size, size+1))
    # turn them into sets of coordinates which populate the grid
    rArraySites = np.vstack([x.flatten(), y.flatten()]).T
    # initialize
    kappaArray = np.zeros_like(x,dtype=float)  #x is integers so we need to cast it to float
    for site in rArraySites:
        kappaArray[site[0]+size,site[1]+size] = calcKappaVVec(site, alpha, v)
    return rArraySites, kappaArray


# def kappaOfR(rArraySites, kappaArray):
#     """ turn grid of kappa(vec r) into kappa(r) """
#     data = []
#     for site in rArraySites:
#         data.append([np.linalg.norm(site), kappaArray[site[0],site[1]]])
#     data = np.array(data)
#     avgs = []
#     # now go through and find places where all rs are equal and calc the avg kappa
#     for r in data[:,0]:
#         avgs.append(np.mean(data[data[:,0] == r,1]))
#     return data[:,0], np.array(avgs)

def kappaOfRFixed(rArraySites, kappaArray,size):
    """ trn grid of kappa(vec r) into kappa(r)"""
    data = []
    for site in rArraySites:
        data.append([np.linalg.norm(site), kappaArray[site[0]+size,site[1]+size]])
    data = np.array(data)
    avgs = []
    uniqueR = np.unique(data[:,0])
    for r in uniqueR:
        indices = (data[:,0] == r)
        avgs.append(np.mean(data[indices,1]))
    return data, uniqueR, avgs

# the following functions are to maybe deal with the weirdness at the corners
# by creating a grid and then averaging over all sites with distance r, we're actually
# missing some spots because a square lattice won't have every site that has distance r for the
# corners of the square. 
def find_two_squares(target):
    """ this algorithm finds all coordinates in an octant that would fall on a circle"""
    pairs = []
    # left pointer starts at 0, right pointer starts at the sqrt of the target
    left = 0
    right = int(math.sqrt(target))
    while left <= right:
        current_sum = left**2 + right ** 2
        if current_sum == target:
            pairs.append(left, right)
            left += 1
            right -= 1
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return pairs