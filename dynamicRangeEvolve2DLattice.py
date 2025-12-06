import numpy as np
import os
from time import time as wallTime
from numba import njit
from randNumberGeneration import getRandomDistribution
import json
from datetime import date
import h5py
import sys
import shutil


# started 1 Dec 2025 to replace memEfficientEvolve2DLattice and the diffusionND modules
# the goal of this is to use the numba code form memEfficientEvolve2DLattice
# and combine it with the directed polymer precision scheme to make the most out of
# dynamic range. we store the occupancy in the form of log(occ).
# this is to avoid quads and c++
# note: we are doing a minimal version with the most specific posibile choices of RWRE

def moveProbabilityFromSite(logP, i, j, biases):
    """ procedure to move probability from site i,j to its nearest neighbor sites, stored as logP, using biases"""
    # iterate over direcitons... i +-1, j+- 1?
    directions = [(i, j - 1), (i + 1, j), (i - 1, j), (i, j + 1)]
    for index, direction in enumerate(directions):
        # if the adjcent site is unoccupied (filled with -infs)
        logTransitionProb = np.log(biases[index]) + logP[i, j]  # ln(xi*p) = ln(xi) + ln(p)
        if np.isneginf(logP[direction]):
            # update the unoccupied adjacent site with (ln(xi) + ln(P[i,j])
            logP[direction] = logTransitionProb
        else:  # if the adjacent site is occupied with some ln(P(direction))
            smallLog, bigLog = np.sort([logTransitionProb, logP[direction]])
            # this is the precision scheme
            logP[direction] = bigLog + np.log(1 + np.exp(smallLog - bigLog))
    # reset site to unoccupied
    logP[i, j] = -np.inf
    return  # return tells it when the proecedure is done

def updateLogOccupancy(logP, time):
    """
    update occupancy (stored as logP) from time t-1 to time t using precision scheme from DPRM
    logP: np array (2L+1,2L+1) filled with -infs(unoccupied) and log(p) values
    time: int, time at which occupancy is being updated to
    """
    L = logP.shape[0] // 2
    if time < (L-1):  # shrinkwrap array scan
        start = L - time  # if t = 9999, 10000 - 9999 = 1
        end = L + time  # if t= 09999, 10000 + 9999 = 19999
    else:  # start at 1 and end at 2L to preserve final square as absorbing boundary
        start = 1
        end = logP.shape[0] - 1 # the last index is 2L-1 (the absorbing boundary) and setting the end will stop the range 1 before that
    # iterate over current state of the array, only occupied sites
    for i in range(start, end):
        for j in range(start, end):
            if (i + j + time) % 2 == 1:
                biases = np.random.dirichlet([1]*4)  # draw set of biases for site i,j to move
                # update logP arary using precision scheme for each direction
                moveProbabilityFromSite(logP, i, j, biases)
    return logP

def logOccupancyGenerator(logOccupancy, maxT, startT=1):
    """ generator for updateLogOccupancy """
    for t in range(startT, maxT):
        logOccupancy = updateLogOccupancy(logOccupancy, t)
        yield t, logOccupancy

def sumLogList(logArray):
    """ implements DPRM precision scheme for adding small numbers, given a 1d array of numbers to be addeed"""
    # return logList[bigIndex] + np.sum(exp(logList - logList[bigIndex))
    bigIndex = np.argmax(logArray)
    return logArray[bigIndex] + np.log( np.sum( np.exp(logArray - logArray[bigIndex] ) ) )

def sumOctantLog(logOccupancy, i, j):
    """ given an array of logP values (logOccupancy), using site (i,j) find all the octant symmetries and sum them """
    L = logOccupancy.shape[0] // 2
    # Origin i == j == L
    if (i == L) and (j == L):
        print('Origin')
        return logOccupancy[i,j]
    # Diagonal i == j
    elif (i == j):
        print('Diagonal')
        return sumLogList(np.array([logOccupancy[i,j], logOccupancy[i,2*L-j], logOccupancy[2*L-i,2*L-j], logOccupancy[2*L-i,j]]))
    # Cardinal i == L
    elif (i == L):
        print('Cardinal')
        return sumLogList(np.array([logOccupancy[j,L], logOccupancy[2*L-j,L], logOccupancy[L,j], logOccupancy[L,2*L-j]]))
    # General
    else:
        print('General')
        return sumLogList(np.array([logOccupancy[i,j], logOccupancy[i,2*L-j], logOccupancy[2*L-i,2*L-j], logOccupancy[2*L-i,j],
                                    logOccupancy[j,i], logOccupancy[2*L-j,i], logOccupancy[2*L-j,2*L-i], logOccupancy[j,2*L-i]]))

def measureProbabilityPastCircle(logOccupancy, radiiListSq, time):
    """
    given a logOccupancy at a time, meeasure the sum of probabilities past a
    circle with given radii. This does this via a cumulative sum past each radius
    logOccupancy: np array, stored with values of logP_site
    radiiListSq: array of floats describing radii of circles past which measurement made (should be
        just a 1d array of length velocities). stores the r^2 values specifically. I think this needs to be ordered smallest to largest
    time: int, time at which the measurements of sum of probability past circle are being taken

    returns:
    cumLogProbList: array that matches size of radiiListSq. Each entry is the

    NOTE: This algorithm prioritizes memory over speed.  If we wanted the opposite we would
    pass in an array with precomputed distances instead.
    """
    # setup/initialization
    cumLogProbList = np.full_like(radiiListSq, -np.inf)
    L = logOccupancy.shape[0] // 2
    if time < (L-1):  # shrinkwrap array scan; end val. only needed for one octant
        end = L + time  # if t= 09999, 10000 + 9999 = 19999
    else:  # for the measurement we WANT the absorbing boundary, so no -1
        end = logOccupancy.shape[0]
    # First, compute the squared distance to every relevant pixel
    # Only compute distances within an octant, and start and the origin and work outward
    for i in range(L, end):
        for j in range(i, end):
            # Only look at occupied sites, which we can check by looking for finite vals
            if np.isfinite(logOccupancy[i,j]):
                distSq = (i-L)**2 + (j-L)**2
                # compute the sum of every point along the circle given by (i,j)?
                # this works because of circular symmetry
                sumRepeats = sumOctantLog(logOccupancy, i, j)  # this function contains the octant logic
                print(i, j, logOccupancy[i, j], sumRepeats)
                # now build up cumulative sum as each measurement r gets bigger and bigger
                for index, rSq in enumerate(radiiListSq):
                    # if the site i,j is greater than the r_measurement, add it to that corresponding index
                    # in cumLogProbList
                    if distSq > rSq:
                        print("distSq, rSq",distSq, rSq)
                        cumLogProbList[index] = sumLogList(np.array([sumRepeats, cumLogProbList[index]]))
    return cumLogProbList
