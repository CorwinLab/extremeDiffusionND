import numpy as np
import os
# import scipy.stats as ss
# from scipy.ndimage import morphology as m
# import csv
# import npquad
import pandas as pd
import sys
# import glob
from numba import njit, vectorize


#from evolve2DLattice import getListOfTimes


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
    probability = np.zeros_like(distances)  # distance should be passed in as radiiList[:,timestep,:]
    origin = [occupancy.shape[0] // 2, occupancy.shape[1] // 2]
    # iterate over the current occupancy
    for i in range(0, occupancy.shape[0]):
        for j in range(0, occupancy.shape[1]):
            # iterate over the the radii that are passsed in
            for k in range(distances.shape[0]):
                for l in range(distances.shape[1]):
                    # check if past barrier for each radius, and if so, add the prob. from that site
                    if np.square(distances[k, l]) >= np.square(i - origin[0]) + np.square(j - origin[1]):
                        probability[k, l] += occupancy[i, j]
    return probability


# i need a 3-index np array, time, velocity, scaling
def calculateRadii(times, velocity, scalingFunction):
    """
    get list of radii = v*(function of time) for barrier for given times
    Ex: radii = calculateRadii(np.array([1,5,10]),np.array([[0.01,0.1,0.5]]),tOnLogT)
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


# # TODO: no, use npz and put the save inside the for t, occ in evolve2DDirichlet
# and at each timestep you basicaly overwrite the file?
# # also pre-allocate memory for the probabilities file
# also this should be the stuff that was originally in the part of
# measureDirichletPastBox, starting with the with statement
# also this doesn't need to be in numba because it's calling things already made fast by numba
def evolveAndMeasurePDF(ts, tMax, occupancy, radiiList, alphas, saveFile):
    # pre-allocate memory for probability
    probabilityFile = np.zeros_like(radiiList)  # should inherit the shape (#scalings, #times, #velocities)
    # data generation
    for t, occ in evolve2DDirichlet(occupancy, tMax, alphas):
        if t in ts:
            # take measurements
            probs = integratedProbability(occ, radiiList[:, list(ts).index(t), :])
            probabilityFile[:, t, :] = probs  # shape: (# scalings, # velocities)
            # save. note that this overwrites the file each time
            # structure is (scaling, times, velocities)
            np.save(saveFile, probabilityFile)


# mem efficient versino of runQuadrantsData.py???
# Check for files, set everything up
# set up occ and get list of ts, then calculate radii
def runDirichlet(L, tMax, alphas, saveFile, systID):
    # setup
    # this assumes alphpa1=alpha2=alpha3=alpha4 which is ok because that's what we're working with\
    alphas = np.array([alphas] * 4)
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    ts = getListOfTimes(1, tMax)  # array of times
    # TODO: fix velocity calc. to reflect the "want velocities kinda close to 1 but not quite at 1?
    velocities = np.array(
        [np.geomspace(10 ** (-3), 10, 21)])  # the extra np.array([]) outside is to get the correct shape
    # get list of radii, scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
    listOfRadii = np.array([calculateRadii(ts, velocities, linear), calculateRadii(ts, velocities, np.sqrt),
                            calculateRadii(ts, velocities, tOnLogT), calculateRadii(ts, velocities, tOnSqrtLogT)])

    # check if savefile exists already and is complete?

    os.makedirs(saveFile, exist_ok=True)
    actualSaveFile = os.path.join(saveFile, str(systID))
    if os.path.exists(actualSaveFile):
        data = pd.read_csv(actualSaveFile)
        max_time = max(data['Time'].values)
        if max_time == ts[-2]:
            print(f"File Finished", flush=True)
            sys.exit()
    # actually run and save data
    evolveAndMeasurePDF(ts, tMax, occ, listOfRadii, alphas, actualSaveFile)


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
