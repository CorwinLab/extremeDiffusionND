import numpy as np
import os
from time import time as wallTime
from numba import njit
import json
# import tracemalloc
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

    note: generating biases inside the double for loop prioritizes memory. to prioritize
    speed, pre-generate all the biases
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
    # logBiasesAll = np.log(
    #     np.random.dirichlet([1] * 4,
    #                         size=(((end - start + 1) // 2), ((end - start + 1) // 2))))
    # iterate over current state of the array, only occupied sites
    for i in range(start, end):
        for j in range(start, end):
            if (i + j + time) % 2 == 1:
                logBiases = np.log(np.random.dirichlet([1]*4))
                # logBiases = logBiasesAll[
                #     (i - start) // 2, (j - start) // 2, :]  # pull out the set of 4 logBiases for site i,j
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

# # TO DO: i need up, right, and down cardinal directions ONLY at the origin
# # but then I need the caridnal directions all the way out to time whatever.. so idk.
# @njit
# def sumPastLineLog(logOccupancy, i, j):
#     """ procedure. given an array of log prob values (logOccupancy),
#     using site (i,j), find the symmetries across the line located at some r x_hat
#     ie. we want to the right of some vertical line placed at r>L"""
#     L = logOccupancy.shape[0] // 2
#     # oriign i == j == L, need to sweep the y-axis
#     print(f"i, j: {i},{j}")
#     if (i == L) and (j == L):
#         print("origin")
#         return logOccupancy[i,j]
#     # diagonal i == j
#     elif (i == j):
#         print("diagonal")
#         print(f"2L-i, j: {2*L-i},{j}")
#         return sumLogList(np.array([logOccupancy[i,j],
#                                     logOccupancy[2*L-i, j]]))
#     # cardinal i == L: (along +xhat)
#     elif (i == L):  # no symmetry either because we only want 1 cardinal direction
#         print("x axis (i = L)")
#         return logOccupancy[i, j]
#     # general
#     else:  # there should be 4 octants to the right of the vertical line at rx_hat
#         print("general")
#         print(f"j,i: {j},{i} \n 2L-i, j: {2 * L-i},{j} \n 2L-j, i: {2 *L- j},{i}")
#         return sumLogList(np.array([logOccupancy[i, j],
#                                     logOccupancy[j, i],
#                                     logOccupancy[2*L-i, j],
#                                     logOccupancy[2*L-j, i]
#                                     ]))

# if i = L and j is along the + x-axis
# this gets its own measureProbabilityPastShape because it just iterates past the
# x axis and not the entire logOcc array?
@njit
def measureProbabilityPastClosestApproach(logOccupancy, measListSq, time):
    """ given a logOccupancy at some time, measure the sum of values on the x-axis past
    points at x^2 = measListSq. This is done via cumulative some past each distance
    logOccupancy: np array stored with valeus of logP_site
    measListSq: 1d array of floats describing x^2 = r^2 at which measurements made
    time: int time at which measurements made at
    returns:
    cumLogProbList: array that matches measListSq in size
    """
    cumLogProbList = np.full_like(measListSq, -np.inf)
    L = logOccupancy.shape[0] // 2
    if time < (L-1):
        end = L+time
    else:
        end = logOccupancy.shape[0]
    # iterate over the x-axis and take measurements past diff. points. going from smallest to largest dist. away from origin
    # the x axis is occ[L, :]
    for j in range(L, end):  # x axis is L,: so we only iterate over the second index
        if np.isfinite(logOccupancy[L, j]):
            distSq = (j - L)**2  # compute dist. from x-origin i = L so L - L = 0
            for index, rSq in enumerate(measListSq):
                if distSq >= rSq:
                    # TODO: check if this is right!!!!
                    cumLogProbList[index] = sumLogList(np.array([logOccupancy[L,j],cumLogProbList[index]]))
    return cumLogProbList


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
    cumLogProbList: array that matches size of radiiListSq.

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


@njit
def measureProbabilityPastLine(logOccupancy, measListSq, time):
    """ given a logOccupancy at some time, measure the sum of logProbs past a lines
    at x^2 = measListSq. This is done via a cumulative sum past each distance
    logOccupancy: np array stored with values of logP_site
    measListSq: 1d array of floats describing x^2 = r^2 where each measurement is made
    time: int, time at which measurement is made.
    returns:
    cumLogProbList: array that matches size of measListSq
    """
    cumLogProbList = np.full_like(measListSq, -np.inf)
    L = logOccupancy.shape[0] // 2
    if time < (L -1):
        start = L - time
        end = L + time
    else:
        start = 1  # boundary conditions
        end = logOccupancy.shape[0]  # this should be 2L?
    for i in range(start, end):  # iterate from height L - t to L + t
        for j in range(L, end):  # iterate from x=L to x = 2L ?
            if np.isfinite(logOccupancy[i, j]):
                distSq = (j - L)**2 # x-axis distance
                for index, rSq in enumerate(measListSq):
                    if distSq >= rSq:  # at or past line
                        # we use logOccupancy[i,j] instead of sumRepeats here because there's no octant symmetry
                        cumLogProbList[index] = sumLogList(np.array([logOccupancy[i,j], cumLogProbList[index]]))
    return cumLogProbList


def saveLogOccupancyAndTime(logOccFileName, logOccTimeFileName, logOcc, time):
    """"
    process to save the logOcc array at time t using npz compressed
    note: on talapas, the filename should be /scratch/jamming/...etc../sysid/occupancy ? (.npz gets added autmatically)
    """
    np.save(logOccFileName, logOcc)
    np.save(logOccTimeFileName, time)
    return


def loadLogOccupancyAndTime(logOccFileName, logOccTimeFileName):
    """ return a time and LogOcc so that evolution is restartable """
    logOcc = np.load(logOccFileName)
    time = np.load(logOccTimeFileName)
    return time, logOcc


def saveCumLogProb(cumLogProbFileName, cumLogProb):
    """ process to save the cumulative logProb measurements """
    np.save(cumLogProbFileName, cumLogProb)
    return


def evolveAndMeasure(logOccFileName, logOccTimeFileName, cumLogProbFileName, finalCumLogProbFileName,
                     cumLogProbList, logOcc, rSqArray, times, saveInterval, startT=1,
                     measurement='circle', pointCumLogProbList=None, pointCumLogProbFileName=None, finalPointCumLogProbFileName=None):
    """
    process; given array of logOccupancy (occupancy stored as logOcc vals) and list of times t and list of measurement radii rSqList
    return cumulative logProb array of shape (num of times, num of radii)
    saves every "saveInterval" hours
    """
    # return an array of cumLogProbList which is an unstructured array
    # that corresponds to r's and t's
    startWallTime = wallTime()
    if np.any(times < 1):  # never encounter t < 1.
        raise ValueError("t < 1 included in list of times.")
    for t, occ in logOccupancyGenerator(logOcc, max(times) + 1, startT=startT):  # time evolve using the generator
        if t in times:  # if at a measurement time
            # print(t)
            tIndex = np.where(times == t)[0][0]
            # grab radii that correspond to that time, should be a 1d slice
            radiiAtTimeT = rSqArray[tIndex, :]
            # to deal with regimes if we want them
            sortIndices = np.argsort(radiiAtTimeT)  # it's convenient to have the radii sorted when we make the measurement
            reverseIndices = np.argsort(sortIndices)  # but we want to save them in the original order they were in
            if measurement == 'circle':  # circle measurements.
                sortedCircleMeasurements = measureProbabilityPastCircle(logOcc, radiiAtTimeT[sortIndices],t)
                cumLogProbList.append(sortedCircleMeasurements[reverseIndices])
            elif measurement == 'line':  # line and (maybe) point meas.
                sortedLineMeasurements = measureProbabilityPastLine(logOcc, radiiAtTimeT[sortIndices], t)
                cumLogProbList.append(sortedLineMeasurements[reverseIndices])
                if pointCumLogProbList is not None:  # use passing in pointCumLogProb as a flag
                    sortedPointMeasurements = measureProbabilityPastClosestApproach(logOcc, radiiAtTimeT[sortIndices],t)
                    pointCumLogProbList.append(sortedPointMeasurements[reverseIndices])
        if (wallTime() - startWallTime >= (saveInterval * 3600)):  # save every interval (hrs)
            startWallTime = wallTime()  # reset timer once checked
            saveLogOccupancyAndTime(logOccFileName, logOccTimeFileName, logOcc, t)  # save occupancy
            saveCumLogProb(cumLogProbFileName, np.array(cumLogProbList))  # save probability file
            print(f"saved logOcc file and cumulativeLogProb array at time {t}")
            if pointCumLogProbList is not None:  # we might not want the point meas. when doing line?
                # hard code in assumed filename structure for past a point
                saveCumLogProb(pointCumLogProbFileName, np.array(pointCumLogProbList))
                print('saved pointCumLogProb array')
    # shape: (num of times, num of radii)
    # Save the measurement and delete the occupancy after evolution
    saveCumLogProb(finalCumLogProbFileName, np.array(cumLogProbList))
    if pointCumLogProbList is not None: # repeat logic for cumLogProb, but use it being None
        saveCumLogProb(finalPointCumLogProbFileName, np.array(pointCumLogProbList))
    print("finished evolving! saved final cumulative probability list")
    if os.path.exists(logOccFileName) and os.path.exists(logOccTimeFileName):
        os.remove(logOccFileName)
        os.remove(logOccTimeFileName)
    if os.path.exists(cumLogProbFileName) and os.path.exists(finalCumLogProbFileName):
        os.remove(cumLogProbFileName)
    print("deleted final occupancy and intermediate cumLogProb file")
    if pointCumLogProbList is not None:
        if os.path.exists(pointCumLogProbFileName) and os.path.exists(finalPointCumLogProbFileName):
            os.remove(pointCumLogProbFileName)
        print("deleted intermediate pointCumLogProb file")
    return


def runSystemCircle(L, velocities, tMax, topDir, sysID, saveInterval):
    """
    initialize occupancy, radii, and times. run evolution of 2D RWRE and save every 3 hrs
    """

    # largest = np.max([L, sysID, tMax])  # in case we want reproducible random numbers
    # we start 1 to tmax instead of 0 to tmax-1
    times = np.unique(np.geomspace(1, tMax, num=500).astype(int))
    measurement = 'circle'
    # hard-code in the linearly scaled radii
    radiiSqArray = (velocities * np.expand_dims(times, 1)) ** 2

    # assumes topDir is /projects/jamming/fransces/data/...etc.../
    cumLogProbFileName = os.path.join(topDir, f"{sysID}.npy")
    finalCumLogProbFileName = os.path.join(topDir, "Final"+f"{sysID}.npy")
    print(f"cumLogProbFileName: {cumLogProbFileName}")
    # occupancy & time files go into the scratch directory
    occTopDir = topDir.replace("projects", "scratch")
    os.makedirs(occTopDir, exist_ok=True)  # need to generate the occupancy file paths
    logOccFileName = os.path.join(occTopDir,f"Occupancy{sysID}.npy")
    logOccTimeFileName = logOccFileName.replace("Occupancy", "time")
    print(f"logOccFileName: {logOccFileName}")

    # note: if it fucks up (file doesn't exist, file doesn't read in properly, etc). then let the code fail
    if os.path.exists(finalCumLogProbFileName):  # if cumLogProb file exists and is final
        print("finalCumLogProbFileName exists, evoultion already complete. exiting", flush=True)
        return
    else:  # otherwise evolve
        if os.path.exists(logOccFileName) and os.path.exists(logOccTimeFileName) and os.path.exists(cumLogProbFileName):  # continue evolution
            # if logOcc File exists
            print("existing logOcc", flush=True)
            currentTime, logOcc = loadLogOccupancyAndTime(logOccFileName, logOccTimeFileName)
            print(f"loaded from {currentTime}", flush=True)
            cumLogProbList = list(np.load(cumLogProbFileName))
            # Reload state of random number generator ? if we want reproducible random numbers
        else:  # start from scratch and initialize from t=0
            # seed = L * largest ** 2 + tMax * largest + sysID  # for reproducible random numbers
            cumLogProbList = []
            logOcc = np.full((2 * L + 1, 2 * L + 1), -np.inf)
            logOcc[L, L] = np.log(1)  # 0
            currentTime = times[0]
        # run evolution and saving
        evolveAndMeasure(logOccFileName, logOccTimeFileName, cumLogProbFileName, finalCumLogProbFileName, cumLogProbList, logOcc, radiiSqArray, times,
                     saveInterval=saveInterval, startT=currentTime, measurement=measurement)
        return  # end of runSystem process


    # this includes the regimes
def runSystemLine(L, velocities, tMax, topDir, sysID, saveInterval):
    """
    process
    initialize occupancy, radii, and times. run evolution of 2D RWRE and save every 3 hrs
    this is to make the past a line AND the past the point of closest approach measurement
    AND implements the vt, vt/sqrt(ln(t)), and vt^1/2 regimes again
    """
    # we start 1 to tmax instead of 0 to tmax-1
    times = np.unique(np.geomspace(1, tMax, num=500).astype(int))
    measurement = 'line'
    # prep measuremnt distances for each regime
    sqrtRadii = (velocities * np.expand_dims(np.sqrt(times), 1)) ** 2
    criticalRadii = (velocities * np.expand_dims(times/np.sqrt(np.log(times)), 1))**2
    linearRadii = (velocities * np.expand_dims(times, 1)) ** 2
    # this should be a # oft by 3*# ofvelocities 2d array
    radiiSqArray = np.hstack((sqrtRadii, criticalRadii, linearRadii))

    # assumes topDir is /projects/jamming/fransces/data/...etc.../Line/
    cumLogProbFileName = os.path.join(newTopDir, f"{sysID}.npy")
    finalCumLogProbFileName = os.path.join(newTopDir, "Final"+f"{sysID}.npy")
    print(f"cumLogProbFileName: {cumLogProbFileName}", flush=True)
    # .../L$L/LINE/Point0.npy or .../L$L/LINE/FinalPoint0.npy
    pointCumLogProbFileName = cumLogProbFileName.replace(f"{sysID}.npy", f"Point{sysID}.npy")
    finalPointCumLogProbFileName = os.path.join(newTopDir, "Final"+f"Point{sysID}.npy")

    # occupancy & time files go into the scratch directory
    # /scratch/jamming/fransces/data/.../L$L/LINE/...
    occTopDir = newTopDir.replace("projects", "scratch")
    os.makedirs(occTopDir, exist_ok=True)  # need to generate the occupancy file paths
    # /scratch/jamming/fransces/data/.../L$L/LINE/LineOccupancy0.npy
    logOccFileName = os.path.join(occTopDir,f"LineOccupancy{sysID}.npy")
    # /scratch/jamming/fransces/data/.../L$L/LINE/LineTime0.npy
    logOccTimeFileName = logOccFileName.replace("LineOccupancy", "LineTime")
    print(f"logOccFileName: {logOccFileName}", flush=True)

    if os.path.exists(finalCumLogProbFileName) and os.path.exists(finalPointCumLogProbFileName):
        # if cumLogProb file exists and is final, if pointCumLogProb exists and is final
        print("finalCumLogProbFileName and finalPointCumLogProbFileName exist, evoultion already complete. exiting",flush=True)
        return
    else:  # otherwise evolve
        # this should fail if any of these are corrupt
        if os.path.exists(logOccFileName) and os.path.exists(logOccTimeFileName) and os.path.exists(cumLogProbFileName) and os.path.exists(pointCumLogProbFileName):
            # continue evolving if logOcc and both cumLogProbFileName and pointCumLogProbFileName exist
            print("existing logOcc and measurement files",flush=True)
            currentTime, logOcc = loadLogOccupancyAndTime(logOccFileName, logOccTimeFileName)
            print(f"loaded from {currentTime}",flush=True)
            cumLogProbList = list(np.load(cumLogProbFileName))
            pointCumLogProbList = list(np.load(pointCumLogProbFileName))
            # Reload state of random number generator ? if we want reproducible random numbers
        else:  # start from scratch and initialize from t=0
            # seed = L * largest ** 2 + tMax * largest + sysID  # for reproducible random numbers
            cumLogProbList = []
            pointCumLogProbList = []
            logOcc = np.full((2 * L + 1, 2 * L + 1), -np.inf)
            logOcc[L, L] = np.log(1)  # 0
            currentTime = times[0]
        # run evolution and saving
        evolveAndMeasure(logOccFileName, logOccTimeFileName, cumLogProbFileName, finalCumLogProbFileName,
                         cumLogProbList, logOcc, radiiSqArray, times,saveInterval=saveInterval, startT=currentTime, measurement=measurement,
                         pointCumLogProbList=pointCumLogProbList, pointCumLogProbFileName=pointCumLogProbFileName,finalPointCumLogProbFileName=finalPointCumLogProbFileName)
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
