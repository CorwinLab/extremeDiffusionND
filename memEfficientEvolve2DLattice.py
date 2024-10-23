import numpy as np
import os
from time import time as wallTime  # start = wallTime() to avoid issues with using time as variable
# import scipy.stats as ss
# from scipy.ndimage import morphology as m
# import csv
# import npquad
import pandas as pd
import sys
import glob
from numba import njit, vectorize


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


@njit
def updateOccupancy(occupancy, time, alphas):
    """
    memory efficient version of executeMoves from evolve2DLattice
    :param occupancy: np array of size L
    :param time: the timestep at which the moves are being executed (int)
    :param alphas: np array, list of 4 numbers > 0 for Dirichlet distribution
    """
    origin = [occupancy.shape[0] // 2, occupancy.shape[1] // 2]
    # for short times, only loop over the part of the array we expect to be occupied
    #TODO: at some point implement "find indx of farthest occupied site
    # and sweep over that square instead
    if time < origin[0]:  # this has an upper limit since vt grows linearly with t.
        startIdx = origin[0] - time
        endIdx = origin[0] + time
    else:
        # start at 1 and go to shape-1 because otherwise at the boundary which we don't want
        # this is an effective way of implementing absorbing boundary conditions
        # note that the boundary is square and not circular
        startIdx = 1
        endIdx = occupancy.shape[0] - 1
    for i in range(startIdx, endIdx):  # down
        for j in range(startIdx, endIdx):  # across
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
                    # <= because we have we want (distToCenter >= radii) to get outside sphere
                    # and the line below has (radii <= dist)
                    if np.square(distances[k, l]) <= np.square(i - origin[0]) + np.square(j - origin[1]):
                        probability[k, l] += occupancy[i, j]
    return probability


#TODO: try/except to suppress warning

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
    return np.unique(np.geomspace(startT, maxT, num=num).astype(int))


# def saveOccupancyState(occ, t, saveFile):
#     """docstring
#     :param occ: np array, the occupancy array to be saved
#     :param t: int, the time at which occupancy is being saved
#     :param saveFile: string, the topDirectory where the probability measurements
#         from evolveAndMeasurePDF are being saved
#     """
#     # generate states directory?
#     statesPath = os.path.join(saveFile+"states")  #directory/sysIDstates
#     os.makedirs(statesPath, exist_ok=True)
#     currentStatePath = os.path.join(statesPath, str(t))  # path/systemIDstates/time
#     # todo: delete everyhing below this line & shove it into updateSaveFile
#     np.savez_compressed(currentStatePath, occ)  # path/systemIDstates/time.npz
#     # now delete the old state
#     files = os.listdir(statesPath)  # get list of states
#     #TODO: make "deleteState" a new function?
#     #TODO: fix this because i need to be really carefully about how many states i have saved
#     if len(files) > 1:  # if there's more than the file you just saved
#         # careful because it assumes there's always only 2 files
#         idx = files.index(f"{str(t)}.txt")  # get index of file you just created
#         files.pop(idx)  # remove it from list so you grab the other file
#         fileToRemove = files[0]  # pull out of list
#         os.remove(os.path.join(statesPath, fileToRemove))  # actually remove the other file
#     #TODO: delete file if completely done

# TODO: change so it doesn't do statePath and os.makedirs inside
# save occupancy at time t
def saveOccupancyState(occ, time, saveFile, temp=True):
    # saveFile is just directory, like "/home/fransces/Documents/code/extremeDiffusionND/temp"
    # generate states directory
    statesPath = os.path.join(saveFile + "states")  #directory/sysIDstates
    #os.makedirs(statesPath, exist_ok=True)
    if temp:
        # path/systemIDstates/time.temp.
        currentStatePath = os.path.join(statesPath, str(time)+".temp")
    else:
        # path/systemIDstates/time
        currentStatePath = os.path.join(statesPath, str(time))
    # saves either path/systemIDstates/time.temp.npz or path/systemIDstates/time.npz
    np.savez_compressed(currentStatePath, occ)  # assumes arr_0 is label


# delete a given stateFile
def deleteOccupancyState(fileName):
    # fileName should be directory/sysIDstates/time.npz
    # delete a single state file: directory/sysIDstates/time.npz
    os.remove(fileName)


#TODO: replace saveOccupancyState with this
# implement atomic whatever here
# this goes inside evolveAndMeasure
def updateSavedState(occ, t, tMax, saveFile):
    """
    there should only ever be 0 or 1 file insie the directory "saveFile+states"
    if 0 then a state is being saved for the first time, so # files 0 --> 1
    if 1 then there was an old state and a new state is being written
        old file exists (# file = 1)
        save new file to temporary name (# files 1 --> 2)
        rename new file from temp to fll name ( # files = 2)
        delete old file (# files 2 --> 1)
    if t = tmax:
        delete saved state(s) (files 1 or 2 --> 0)
    """
    # saveFile is directory, i.e. /home/fransces/Documents/code/temp/sysID or something
    statesPath = os.path.join(saveFile + "states")  #directory/sysIDstates
    # there should only ever be 0 or 1 file in this directory when this function is called
    os.makedirs(statesPath,exist_ok=True)
    files = os.listdir(statesPath)
    print(f"files @ start of updateSavedState: {files}")
    if len(files) == 0:
        print(f"no states, saving now")
        saveOccupancyState(occ, t, saveFile)
    # if file completely finished, delete saved states
    if t == (tMax - 1):  # because of the way python indexes
        print(f"t = {t}, deleting {files}")
        for file in files:
            deleteOccupancyState(os.path.join(statesPath,file))
        # todo: delete 0states directory too
    else:
        # if existing file (say oldtime.npz)
        if len(files) == 1:
            # if files = ['oldTime.npz'] then statesPathOldTime = directory/sysIDstates/oldtime.npz
            statesPathOldTime = os.path.join(statesPath,files[0])
            # directory/systemIDstates/time.temp.npz
            tempFileName = os.path.join(statesPath, str(t)+".temp.npz")
            saveOccupancyState(occ, t, saveFile, temp=True)
            print(f"temp file saved {tempFileName}")
            # rename to path/systemIDstates/time.npz, to indicate completion
            completedFileName = os.path.join(statesPath, str(t)+".npz")
            os.rename(tempFileName, completedFileName)
            print(f"completed file {completedFileName}")
            # delete oldtime.npz
            deleteOccupancyState(statesPathOldTime)
            print(f"temp file deleted")
    # if filecompletely finished, delete any/all saved states
    # if t == (tMax-1):
        # TODO: fix because I think depending on where this go    # f file completely finished, delete any/all saved states
    # if t == (tMax-1):
        # TODO: fix because I think depending on where this goes
        #  the state named finalTime.npz will not have been saved...                                                            # delete time.npz
        # deleteOccupancyState(statesPath, t)
        # write like this in case there happens to be a temp file?? idk
        # for file in files:
            # remove directory/sysIDstates/filesInStates.npz
            # deleteOccupancyState(file)es
        #  the state named finalTime.npz will not have been saved...
        # delete time.npz
        # deleteOccupancyState(statesPath, t)
        # write like this in case there happens to be a temp file?? idk
        # for file in files:
            # remove directory/sysIDstates/filesInStates.npz
            # deleteOccupancyState(file)


# TODO: this goes in runDirichlet, before evolveAndMeasure
def restoreOccupancyState(statesPath):
    """
    :param statesPath: full path to where states are saved, directory/sysIDstates
    """
    # old way
    # # load in occupancy array
    # states = os.listdir(statesPath)  # i think this will fail if there are no states..
    # occ = np.load(f"{statesPath}/{states[0]}")['arr_0']  # load back in saved state. assumes no keyword
    # t = int(states[0][:-4])  # assumes filename is (int).npz
    # return t, occ

    # new logic: if tehre's a temp file then restore the non-temp file
    # TODO: carefull, assumes only 1 non-temp file
    files = os.listdir(statesPath)
    # if there's temporary files then restore the old file
    tempFiles = glob.glob("*temp.npz", root_dir=statesPath)
    if len(tempFiles) != 0:
        for file in tempFiles:
            # delete any temporary file(s)
            deleteOccupancyState(os.path.join(statesPath, file))
            # and remove it from the list of files
            files.remove(file)
            # by this point there should only be one 1 entry in files
    # otherwise there should only be one file time.npz
    occ = np.load(f"{statesPath}/{files[0]}")["arr_0"]
    t = int(files[0][:-4])  # assumpes (time).npz
    return t, occ


#TODO!!: write in shit to ignore unfinished files
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
    # grab the files in the data directory, ignoring subdirectories
    # files = os.listdir(path)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    if 'info.npz' in files:
        files.remove('info.npz')
    times = np.load(f"{path}/info.npz")['times']
    # initialize the moments & mask, fence problem
    firstData = np.load(f"{path}/{files[0]}")
    # if given some tCutoff:
    if tCutOff is not None:
        # find index val. of the closest tCutoff. this is a dumb way to do that
        idx = list(times).index(times[times < tCutOff][-1])
        # cut all data off to that val
        firstData = firstData[:, :idx + 1, :]
    if takeLog:
        firstData = np.log(firstData)
    moment1, moment2, moment3, moment4 = firstData, np.square(firstData), np.power(firstData, 3), np.power(firstData, 4)
    # load in rest of files to do mean var calc, excluding the 0th file
    for file in files[1:]:
        data = np.load(f"{path}/{file}")
        if tCutOff is not None:  # only chop data if you give it a cutoff time
            data = data[:, :idx + 1, :]
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
    startTime = wallTime()
    # data generation
    for t, occ in evolve2DDirichlet(occupancy, tMax, alphas, startT=startT):
        if t in ts:
            idx = list(ts).index(t)
            probs = integratedProbability(occ, radiiList[:, idx, :])  # take measurements
            probabilityFile[:, idx, :] = probs  # shape: (# scalings, # velocities)
            # save. note that this overwrites the file at each time
            # structure is (scaling, times, velocities)
            # scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
            np.save(saveFile, probabilityFile)  # saves to directory/systemID.npy
        # if wallTime() - startTime >= 10800:  # 3 hrs?
        if (wallTime() - startTime >= 30) or (t==tMax-1):  # 1 min
            # this saves to path/systemIDstates/
            # saveOccupancyState(occ, t, saveFile)  #passes in directory/sysID
            updateSavedState(occ, t, tMax, saveFile)
            startTime = wallTime()  # reset wallTime for new interval


# mem efficient versino of runQuadrantsData.dpy???
# Check for files, set everything up
# set up occ and get list of ts, then calculate radii]]
def runDirichlet(L, tMax, alphas, saveFile, systID):
    # setup
    # assumes alphpa1=alpha2=alpha3=alpha4
    alphas = np.array([alphas] * 4)
    # check thtat there is a state path that isn't empty
    actualSaveFile = os.path.join(saveFile, str(systID))  # directory/systID
    statesPath = os.path.join(actualSaveFile + "states")  # directory/systIDstates
    # only restore states if the states path exists and isn't empty
    if (os.path.exists(statesPath)) and (len(os.listdir(statesPath)) != 0):
        mostRecentTime, occ = restoreOccupancyState(statesPath)
        info = np.load(f"{saveFile}/info.npz")
        ts = info['times']
        velocities = info['velocities']
    # otherwise generate occupancy, times, velocities as normal
    else:
        occ = np.zeros((2 * L + 1, 2 * L + 1))
        occ[L, L] = 1
        mostRecentTime = 1
        # array of times, tMax-1 because we don't want to include the last t
        # this isn't an issue with evolve2Dlattice bc we're initializing probabiityFile differently
        ts = getListOfTimes(tMax - 1, 1)
        velocities = np.array(
            [np.geomspace(10 ** (-3), 10, 21)])  # the extra np.array([]) outside is to get the correct shape
    # get list of radii, scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
    listOfRadii = np.array([calculateRadii(ts, velocities, linear), calculateRadii(ts, velocities, np.sqrt),
                            calculateRadii(ts, velocities, tOnLogT), calculateRadii(ts, velocities, tOnSqrtLogT)])
    # check if savefile exists already and is complete?
    os.makedirs(saveFile, exist_ok=True)
    # if the file exists and is complete, then exit
    if os.path.exists(os.path.join(actualSaveFile+".npy")): # directory/systID.npy
        temp = np.load(f"{os.path.join(actualSaveFile+'.npy')}")
        # get last nonzero element; if finished idx should match the shape of the list of times
        idx = np.max(np.nonzero(temp))
        # tMax shape -1 because of the way python indexes
        if idx == (ts.shape[0] - 1):
            print(f"File Finished", flush=True)
            sys.exit()
    # if an info file doesn't exist, then create one. otherwise just continue with data gen.
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
