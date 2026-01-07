import numpy as np
import os
from time import time as wallTime
from numba import njit
import json
from datetime import date
import sys
import shutil


# started 1 Dec 2025 to replace memEfficientEvolve2DLattice and the diffusionND modules
# the goal of this is to use the numba code form memEfficientEvolve2DLattice
# and combine it with the directed polymer (DPRM) precision scheme to make the most out of
# dynamic range. we store the occupancy in the form of log(occ).
# this is to avoid quads and c++
# note: we are doing a minimal version with the most specific posibile choices of RWRE
# also keep the idea in mind that the functions should have as minimal interface as possible? (pass in the least amount of info possible)

@njit
def moveProbabilityFromSite(logP, i, j, logBiases):
    """
    procedure to move log probability from site i,j to its nearest neighbor sites, using (log) biases
    logP: ocupancy (np) array with probs at each site stored as logPs
    i, j: ints, indices of the site of logP for which log probs. being moved
    logBiases: set of 4 biases, stored as their log
    """
    # iterate over direcitons
    directions = [(i, j - 1), (i + 1, j), (i - 1, j), (i, j + 1)]
    for index, direction in enumerate(directions):
        logTransitionProb = logBiases[index] + logP[i, j]  # ln(xi*p) = ln(xi) + ln(p)
        # if the adjcent site is unoccupied (filled with -infs)
        if np.isneginf(logP[direction]):
            # update the unoccupied adjacent site with (ln(xi) + ln(P[i,j])
            logP[direction] = logTransitionProb
        else:  # if the adjacent site is occupied with some ln(P(direction))
            smallLog, bigLog = np.sort([logTransitionProb, logP[direction]])
            # this is the precision scheme
            logP[direction] = bigLog + np.log(1 + np.exp(smallLog - bigLog))
    # reset current site to unoccupied
    logP[i, j] = -np.inf
    return  # return tells it when the proecedure is done


@njit
def updateLogOccupancy(logP, time):
    """
    update occupancy (stored as logP) from time t-1 to time t using precision scheme from DPRM
    logP: np array (2L+1,2L+1) filled with -infs(unoccupied) and log(p) values
    time: int, time at which occupancy is being updated to
    """
    L = logP.shape[0] // 2
    if time < (L - 1):  # shrinkwrap array scan
        start = L - time  # if t = 9999, 10000 - 9999 = 1
        end = L + time  # if t= 09999, 10000 + 9999 = 19999
    else:  # start at 1 and end at 2L to preserve final square as absorbing boundary
        start = 1
        end = logP.shape[
                  0] - 1  # the last index is 2L-1 (the absorbing boundary) and setting the end will stop the range 1 before that
    # pre-generating the (log) biases for our shrinkwrapped, checkerboarded sub-area
    logBiasesAll = np.log(
        np.random.dirichlet([1] * 4,
                            size=(((end - start + 1) // 2), ((end - start + 1) // 2))))
    # iterate over current state of the array, only occupied sites
    for i in range(start, end):
        for j in range(start, end):
            if (i + j + time) % 2 == 1:
                logBiases = logBiasesAll[
                    (i - start) // 2, (j - start) // 2, :]  # pull out the set of 4 logBiases for site i,j
                # update logP arary using precision scheme for each direction
                moveProbabilityFromSite(logP, i, j, logBiases)
    return logP


def logOccupancyGenerator(logOccupancy, maxT, startT=1):
    """ generator for updateLogOccupancy to evolve from time startT to time maxT"""
    for t in range(startT, maxT):
        logOccupancy = updateLogOccupancy(logOccupancy, t)
        yield t, logOccupancy


@njit
def sumLogList(logArray):
    """ implements DPRM precision scheme for adding small numbers, given a 1d array of numbers (stored as logP) to be addeed"""
    # return logList[bigIndex] + np.sum(exp(logList - logList[bigIndex))
    bigIndex = np.argmax(logArray)
    return logArray[bigIndex] + np.log(np.sum(np.exp(logArray - logArray[bigIndex])))


@njit
def sumOctantLog(logOccupancy, i, j):
    """
    procedure. given an array of log prob. values (logOccupancy), using site (i,j) find all the octant symmetries and sum them
    """
    L = logOccupancy.shape[0] // 2
    # Origin i == j == L
    if (i == L) and (j == L):
        return logOccupancy[i, j]
    # Diagonal i == j
    elif (i == j):
        return sumLogList(np.array([logOccupancy[i, j], logOccupancy[i, 2 * L - j], logOccupancy[2 * L - i, 2 * L - j],
                                    logOccupancy[2 * L - i, j]]))
    # Cardinal i == L
    elif (i == L):
        return sumLogList(
            np.array([logOccupancy[j, L], logOccupancy[2 * L - j, L], logOccupancy[L, j], logOccupancy[L, 2 * L - j]]))
    # General
    else:
        return sumLogList(np.array([logOccupancy[i, j], logOccupancy[i, 2 * L - j], logOccupancy[2 * L - i, 2 * L - j],
                                    logOccupancy[2 * L - i, j],
                                    logOccupancy[j, i], logOccupancy[2 * L - j, i], logOccupancy[2 * L - j, 2 * L - i],
                                    logOccupancy[j, 2 * L - i]]))


@njit
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
    if time < (L - 1):  # shrinkwrap array scan; end val. only needed for one octant
        end = L + time  # if t= 09999, 10000 + 9999 = 19999
    else:  # for the measurement we WANT the absorbing boundary, so no -1
        end = logOccupancy.shape[0]
    # First, compute the squared distance to every relevant pixel
    # Only compute distances within an octant, and start and the origin and work outward
    for i in range(L, end):
        for j in range(i, end):
            # Only look at occupied sites, which we can check by looking for finite vals
            if np.isfinite(logOccupancy[i, j]):
                distSq = (i - L) ** 2 + (j - L) ** 2
                # compute the sum of every point along the circle given by (i,j)?
                # this works because of circular symmetry
                sumRepeats = sumOctantLog(logOccupancy, i, j)  # this function contains the octant logic
                # now build up cumulative sum as each measurement r gets bigger and bigger
                for index, rSq in enumerate(radiiListSq):
                    # if the site i,j is greater than the r_measurement, add it to that corresponding index
                    # in cumLogProbList
                    if distSq >= rSq:
                        # print("distSq, rSq", distSq, rSq)
                        cumLogProbList[index] = sumLogList(np.array([sumRepeats, cumLogProbList[index]]))
    return cumLogProbList


def saveLogOccupancy(logOccFileName, logOcc, time):
    """"
    process to save the logOcc array at time t using npz compressed
    note: on talapas, the filename should be /scratch/jamming/...etc../sysid/occupancy ? (.npz gets added autmatically)
    """
    np.savez_compressed(logOccFileName, logOcc=logOcc, time=time)
    return


def loadLogOcccupancy(logOccFileName):
    """ return a time and LogOcc so that evolution is restartable """
    temp = np.load(logOccFileName)  # keys should be logOcc, time
    return int(temp['time']), temp['logOcc']


def saveCumLogProb(cumLogProbFileName, cumLogProb):
    """ process to save the cumulative logProb measurements """
    np.save(cumLogProbFileName, cumLogProb)
    return


def evolveAndMeasure(logOccFileName, cumLogProbFileName, cumLogProbList, logOcc, rSqArray, times,
                     saveInterval, startT=1,):
    """
    process
    given array of logOccupancy (occupancy stored as logOcc vals) and list of times t and list of measurement radii rSqList
    return cumulative logProb array of shape (num of times, num of radii)
    saves every "saveInterval" hours
    """
    # return an array of cumLogProbList which is an unstructured array
    # that corresponds to r's and t's
    startWallTime = wallTime()
    seconds = saveInterval * 3600  # num of seconds in saveInterval (hours)
    if np.any(times < 1):  # never encounter t < 1.
        raise ValueError("t < 1 included in list of times.")
    for t, occ in logOccupancyGenerator(logOcc, max(times) + 1, startT=startT):  # time evolve using the generator
        if t in times:  # if at a measurement time
            # print(t)
            tIndex = np.where(times == t)[0][0]
            # grab radii that correspond to that time, should be a 1d slice
            radiiAtTimeT = rSqArray[tIndex, :]
            # do the measurement using measureProbabilityPastCircle
            cumLogProbList.append(measureProbabilityPastCircle(logOcc, radiiAtTimeT, t))
        if (wallTime() - startWallTime >= seconds):  # save every 3 hours
            saveLogOccupancy(logOccFileName, logOcc, t)  # save occupancy
            saveCumLogProb(cumLogProbFileName, np.array(cumLogProbList))  # save probability file
            print(f"saved logOcc file and cumulativeLogProb array at time {t}")
    # shape: (num of times, num of radii)
    print(f"run time: {wallTime() - startWallTime}")
    # Save the measurement and delete the occupancy after evolution
    saveCumLogProb(cumLogProbFileName, np.array(cumLogProbList))
    print("finished evolving! saved final cumulative probability list")
    if os.path.exists(logOccFileName):
        os.remove(logOccFileName)
    print("deleted final occupancy")
    return


def runSystem(L, velocities, tMax, topDir, sysID, saveInterval):
    """
    initialize occupancy, radii, and times. run evolution of 2D RWRE and save every 3 hrs
    """
    # largest = np.max([L, sysID, tMax])  # in case we want reproducible random numbers
    # we start 1 to tmax instead of 0 to tmax-1
    times = np.unique(np.geomspace(1, tMax, num=500).astype(int))
    # hard-code in the linearly scaled radii
    radiiSqArray = (velocities * np.expand_dims(times, 1)) ** 2

    # assumes topDir is /projects/jamming/fransces/data/...etc.../
    cumLogProbFileName = os.path.join(topDir, f"{sysID}.npy")
    print(f"cumLogProbFileName: {cumLogProbFileName}")
    # occupancy file goes into the scratch directory
    occTopDir = topDir.replace("projects", "scratch")
    os.makedirs(occTopDir, exist_ok=True)  # need to generate the occupancy file paths
    logOccFileName = os.path.join(occTopDir,f"Occupancy{sysID}.npz")
    # # occupancy naming for debugging
    # logOccFileName = cumLogProbFileName.replace(f"{sysID}.npy", f"Occupancy{sysID}.npz")
    print(f"logOccFileName: {logOccFileName}")

    # note: if it fucks up (file doesn't exist, file doesn't read in properly, etc). then let the code fail
    if os.path.exists(logOccFileName) and os.path.exists(cumLogProbFileName):
        print("existing logOcc")
        # if logOcc File exists
        currentTime, logOcc = loadLogOcccupancy(logOccFileName)
        print(f"loaded from {currentTime}")
        cumLogProbList = list(np.load(cumLogProbFileName))
        # Reload state of random number generator ? if we want reproducible random numbers
    else:  # initialize from t = 0 (where all prob. at origin)
        # seed = L * largest ** 2 + tMax * largest + sysID  # for reproducible random numbers
        cumLogProbList = []
        logOcc = np.full((2 * L + 1, 2 * L + 1), -np.inf)
        logOcc[L, L] = np.log(1)  # 0
        currentTime = times[0]
    # run evolution and saving
    evolveAndMeasure(logOccFileName, cumLogProbFileName, cumLogProbList, logOcc, radiiSqArray, times,
                     saveInterval=saveInterval, startT=currentTime)
    return  # end of runSystem process


def saveVars(variables, save_file):
    """
    Procedure to save experiment variables to a file along with date it was ran
    """
    for key, item in variables.items():
        if isinstance(item, np.ndarray):
            variables[key] = item.tolist()
    with open(save_file, "w+") as file:
        json.dump(variables, file)
    return
