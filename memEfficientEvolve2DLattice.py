import numpy as np
import os
from time import time as wallTime # start = wallTime() to avoid issues with using time as variable
# import scipy.stats as ss
# from scipy.ndimage import morphology as m
# import csv
# import npquad
import pandas as pd
import sys
# import glob
from numba import njit, vectorize


#TODO: rename file to be more specific

# jacob's way of drawing random dirichlet numbers, but isn't needed because
# numba plays nice with np.random.dirichlet (but not rng.dirichlet
@vectorize
def gammaDist(alpha, scale):
    return np.random.gamma(alpha, scale)


@njit
def randomDirichletNumba(alphas):
    """ alphas is an array. there should be the same # of alphas as there are
    biases we're pulling, that is, 4"""
    gammas = gammaDist(alphas, np.ones(alphas.shape))
    return gammas / np.sum(gammas)


#TODO: make specific! not flexible
@njit
def updateOccupancy(occupancy, time, alphas):
    """
    memory efficient version of executeMoves from evolve2DLattice
    :param occupancy: np array of size L
    :param time: the timestep at which the moves are being executed (int)
    :param alphas: np array, list of 4 numbers > 0 for Dirichlet distribution
    """
    # start at 1 and go to shape-1 because otherwise at the boundary which we don't want
    # this is an effective way of implementing absorbing boundary conditions
    for i in range(1, occupancy.shape[0] - 1):  # down
        for j in range(1, occupancy.shape[1] - 1):  # across
            # the following conditions means you're on the checkerboard of occupied sites
            if (i + j + time) % 2 == 1:
                # biases = randomDirichletNumba(alphas)
                biases = np.random.dirichlet(alphas)
                occupancy[i, j - 1] += occupancy[i, j] * biases[0]  # left
                occupancy[i + 1, j] += occupancy[i, j] * biases[1]  # down
                occupancy[i, j + 1] += occupancy[i, j] * biases[2]  # right
                occupancy[i - 1, j] += occupancy[i, j] * biases[3]  # up
                occupancy[i, j] = 0  # zero out the original site
    return occupancy


#@njit
# doesn't need to be in numba because not actually doing anything slow
# it's calling a function that takes time but we've already wrapped that  one in numba
def evolve2DDirichlet(occupancy, maxT, alphas, startT=1):
    """ generator object, memory efficient version of evolve2DLattice
    note that there's no absorbingBoundary because of the way i and j are indexed
    in updateOccupancy
    :param occupancy: np array, inital occupancy
    :param maxT: int, the final time to which system is evolved
    :param alphas: np array of size (4,), the alphas of the dirichlet dist.
    :param startT: optional int default 1, the time at which you want to start evolution
    """
    for t in range(startT, maxT):
        occupancy = updateOccupancy(occupancy, t, alphas)
        yield t, occupancy


# don't use numba?
def dirichletWrapper(*args, **kwargs):
    """ wrapper for evolve2DDirichlet with *args and **kwargs instead """
    for t, occ in evolve2DDirichlet(*args, **kwargs):
        pass
    return t, occ


@njit
def integratedProbability(occupancy, distances):
    """
    calculate the probability past a sphere of radius distance
    effectively the same part os sphere_masks = [getOutsideSphereMask for r in Rs]
    plus probs = [np.sum(occ[mask]) for mask in sphere_masks]
    :param occupancy: np array of growing probability
    :param distances: float or array, radius of sphere past which probability is being measured
    :return probability: float, the integrated/summed probability past radius above
    """
    probability = np.zeros_like(distances)
    origin = [occupancy.shape[0] // 2, occupancy.shape[1] // 2]
    # iterate over the current occupancy
    for i in range(0, occupancy.shape[0]):
        for j in range(0, occupancy.shape[1]):
            # iterate over the the radii that are passsed in
            for k in range(distances.shape[0]):
                for l in range(distances.shape[1]):
                    #<= because we have we want (distToCenter >= radii) to get outside sphere
                    # and the line below has (radii <= dist)
                    if np.square(distances[k, l]) <= np.square(i - origin[0]) + np.square(j - origin[1]):
                        probability[k, l] += occupancy[i, j]
    return probability


#TODO: try/except to suppress warnings

# i need a 3-index np array, time, velocity, scaling
def calculateRadii(times, velocity, scalingFunction):
    """
    get list of radii = v*(function of time) for barrier for given times; returns array of (# times, # velocities)
    Ex: radii = calculateRadii(np.array([1,5,10]),np.array([[0.01,0.1,0.5]]),tOnLogT)
    To get the original velocities, call radiiVT[0,:]
    """
    times = np.expand_dims(times, 1)  # turns ts into a column vector
    return velocity * scalingFunction(times)


def linear(time):
    return time


def tOnSqrtLogT(time):
    return time / np.sqrt(np.log(time))


def tOnLogT(time):
    return time / np.log(time)


def getListOfTimes(maxT, startT=1, num=500):
    """
	Generate a list of times, with approx. 10 times per decade (via np.geomspace), out to time maxT
	:param maxT: the maximum time to which lattice is being evolved
	:param startT: initial time at which lattice evolution is started
	:param num: number of times you want
	:return: the list of times
	"""
    # do maxT-1 because otherwise it includes tMax, which we don't want?
    return np.unique(np.geomspace(startT, maxT, num=num).astype(int))
#TODO: finish writing this
def saveOccupancyState(occ, t, saveFile):
    """docstring
    :param occ: np array, the occupancy array to be saved
    :param t: int, the time at which occupancy is being saved
    :param saveFile: string, the topDirectory where the probability measurements
        from evolveAndMeasurePDF are being saved
    """
    # generate states directory?
    # path/systemIDstates
    os.makedirs(os.path.join(saveFile + "states"), exist_ok=True)
    # path/systemIDstates/time.txt
    statesPath = os.path.join(saveFile + "states", str(t) + '.txt')
    np.savetxt(statesPath, occ)
    # now delete the old state
    files = os.listdir(statesPath)
    if len(files) > 1: # if there's more than the file you just saved
        # careful because it assumes there's always only 2 files
        idx = files.index(statesPath)  # get index of file you just created
        files.pop(idx)  # remove it from list so you grab the other file
        fileToRemove = files[0]  #pull out of list
        os.remove(os.path.join(statesPath, fileToRemove))  # actually remove the other file


def restoreOccupancyState(states):
    """
    :param states: filelist of states path
    """
    # load in occupancy array
    occ = np.loadtxt(states[0])
    t = int(states[0][:-4])  # assumes filename is (int).txt
    return t, occ


#TODO: write in shit to ignore unfinished files
def getMeasurementMeanVarSkew(path, tCutOff=None, takeLog=True):
    """
	Takes a directory filled  arrays and finds the mean and variance of cumulative probs. past various geometries
	Calculates progressively, since loading in every array will use too much memory
	:param path: the path of the directory, /projects/jamming/fransces/data/quadrant/distribution/tMax
	:param filetype: string, 'box', 'hline', 'vline, 'sphere'
	:param tCutOff: default mmax val of time; otherwise the time at which you want to cut off the data to look at
	:return: moment1: the first moment (mean) of log(probabilities)
	:return variance: the variance of log(probabilities)
	:return skew: the skew of log(probabilities)
	"""
    # grab the files in the data directory that are the Box data
    files = os.listdir(path)
    if 'info.npz' in files:
        files.remove('info.npz')
    # initialize the moments & mask, fence problem
    times = np.load(f"{path}/info.npz")['times']
    firstData = np.load(f"{path}/{files[0]}")
    # if given some tCutoff:
    if tCutOff is not None:
        # find index val. of the closest tCutoff. this is a dumb way to do that
        idx = list(times).index(times[times<tCutOff][-1])
        # cut all data off to that val
        firstData = firstData[:, :idx+1, :]
    if takeLog:
        firstData = np.log(firstData)
    moment1, moment2, moment3, moment4 = firstData, np.square(firstData), np.power(firstData, 3), np.power(firstData, 4)
    # load in rest of files to do mean var calc, excluding the 0th file
    for file in files[1:]:
        data = np.load(f"{path}/{file}")
        if tCutOff is not None:  # only chop data if you give it a cutoff time
            data = data[:,:idx+1,:]
        if takeLog:
            data = np.log(data)
        moment1 += data
        moment2 += np.square(data)
        moment3 += np.power(data, 3)
        moment4 += np.power(data, 4)
    moment1 = moment1 / len(files)
    moment2 = moment2 / len(files)
    moment3 = moment3 / len(files)
    moment4 = moment4 / len(files)
    variance = moment2 - np.square(moment1)
    skew = (moment3 - 3 * moment1 * variance - np.power(moment1, 3)) / (variance) ** (3 / 2)
    kurtosis = (moment4 - 4 * moment1 * moment3 + 6 * (moment1 ** 2) * moment2 - 3 * np.power(moment1, 4)) / (
        np.square(variance))
    # Return the mean, variance, and skew, and excess kurtosis (kurtosis-3)
    # note this also will take the mean and var of time. what you want is
    # meanBox[:,1:] to get just the probs.
    np.savez_compressed(os.path.join(path, "stats.npz"), mean=moment1, variance=variance,
                        skew=skew, excessKurtosis=kurtosis - 3)

# this function is what originally went inside the "with" statement
# it doesn't need numba because it's calling things that already use it
def evolveAndMeasurePDF(ts, startT, tMax, occupancy, radiiList, alphas, saveFile):
    # pre-allocate memory for probability
    probabilityFile = np.zeros_like(radiiList)  # should inherit the shape (#scalings, #times, #velocities)
    # data generation
    startTime = wallTime()
    for t, occ in evolve2DDirichlet(occupancy, tMax, alphas, startT=startT):
        if t in ts:
            idx = list(ts).index(t)
            # take measurements
            probs = integratedProbability(occ, radiiList[:, idx, :])
            probabilityFile[:, idx, :] = probs  # shape: (# scalings, # velocities)
            # save. note that this overwrites the file at each time
            # structure is (scaling, times, velocities)
            # scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT

            # saves to path/systemID.npy
            np.save(saveFile, probabilityFile)
        # todo: also need to add in something that checks for an existing state
        if wallTime() - startTime >= 10800:  # 3 hrs?
            # this saves to path/systemIDstates/
            saveOccupancyState(occ, t, saveFile)
            startTime = wallTime()  # reset wallTime for new interval

# mem efficient versino of runQuadrantsData.py???
# Check for files, set everything up
# set up occ and get list of ts, then calculate radii
def runDirichlet(L, tMax, alphas, saveFile, systID):
    # setup
    # this assumes alphpa1=alpha2=alpha3=alpha4 which is ok because that's what we're working with\
    alphas = np.array([alphas] * 4)
    # check thtat there is a state (and only one)
    statesPath = os.path.join(saveFile + "states")
    # only check for states if the path already exists
    if os.path.exists(statesPath):
        states = os.listdir(statesPath)  # i think this will fail if there are no states..
        # only restore states if the states path exists AND it has the correct file number (1)
        if len(states) == 1:
            mostRecentTime, occ = restoreOccupancyState(states)
            info = np.load(f"{saveFile}/info.npz")
            ts = info['times']
            velocities = info['velocities']
    # otherwise generate occupancy, times, velocities as normal
    else:
        occ = np.zeros((2 * L + 1, 2 * L + 1))
        occ[L, L] = 1
        mostRecentTime = 1
        ts = getListOfTimes(tMax - 1,1)  # array of times, tMax-1 because we don't want to include the last t
        # this isn't an issue with evolve2Dlattice bc we're initializing probabiityFile differently
        # TODO: fix velocity calc. to reflect the "want velocities kinda close to 1 but not quite at 1?
        velocities = np.array(
            [np.geomspace(10 ** (-3), 10, 21)])  # the extra np.array([]) outside is to get the correct shape
    # get list of radii, scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
    listOfRadii = np.array([calculateRadii(ts, velocities, linear), calculateRadii(ts, velocities, np.sqrt),
                            calculateRadii(ts, velocities, tOnLogT), calculateRadii(ts, velocities, tOnSqrtLogT)])

    # check if savefile exists already and is complete?
    os.makedirs(saveFile, exist_ok=True)
    actualSaveFile = os.path.join(saveFile, str(systID))  # this is system num.
    # if the file exists and is complete, then exit
    if os.path.exists(actualSaveFile):
        # info = np.load(os.path.join(saveFile, "info.npz"))
        temp = np.load(f"{actualSaveFile}")
        #todo: make sure still works if everything does happen to be 0? idk
        idx = np.max(np.nonzero(temp))  # get last nonzero element of 
        # tMax shape -1 because of the way python indexes
        if idx == (ts.shape[0]-1):
        #if max_time == ts[-2]:
            print(f"File Finished", flush=True)
            # if its done then exit
            sys.exit()

    if not os.path.exists(os.path.join(saveFile, "info.npz")):
        np.savez_compressed(os.path.join(saveFile, "info"), times=ts, velocities=velocities)
    # actually run and save data
    evolveAndMeasurePDF(ts, mostRecentTime, tMax, occ, listOfRadii, alphas, actualSaveFile)


if __name__ == "__main__":
    # these should call sysargv now, or argparse
    L = int(sys.argv[1])
    tMax = int(sys.argv[2])
    alphas = float(sys.argv[3])
    saveFile = sys.argv[4]
    systID = int(sys.argv[5])
    # initialize occ. & get list of t's

    # actually call runDirichlet here, inside the if __name__ == __main__
    # python memEfficientEvolve2DLattice L tMax alphas velocities (None) saveFile
    runDirichlet(L, tMax, alphas, saveFile, systID)
