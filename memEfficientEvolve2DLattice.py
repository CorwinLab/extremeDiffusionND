import numpy as np
import os
# import scipy.stats as ss
# from scipy.ndimage import morphology as m
import csv
# import npquad
import pandas as pd
import sys
# import glob
from numba import njit, vectorize
from sympy.physics.units import velocity

import evolve2DLattice as ev

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
                occupancy[i, j] = 0
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
def integratedProbability(occupancy, distance):
    """
    calculate the probability past a sphere of radius distance
    effectively the same part os sphere_masks = [getOutsideSphereMask for r in Rs]
    plus probs = [np.sum(occ[mask]) for mask in sphere_masks]
    :param occupancy: np array of growing probability
    :param distance: float or array, radius of sphere past which probability is being measured
    :return probability: float, the integrated/summed probability past radius above
    """
    probability = np.zeros_like(distance)  # distance should be passed in as radiiList[:,timestep,:]
    origin = [occupancy.shape[0] // 2, occupancy.shape[1] // 2]
    for i in range(0, occupancy.shape[0]):
        for j in range(0, occupancy.shape[1]):
            # check if past barrier for each radius, and if so, add the prob. from that site
            # for k in range(len(distance)):
            #     if np.square(distance[k]) >= np.square(i - origin[0]) + np.square(j - origin[1]):
            #         probability[k] += occupancy[i, j]

            # iterate over the the radii that are passsed in
            for k in range(distance.shape[0]):
                for l in range(distance.shape[1]):
                    # if we do k in range len(radii)) we iterate over each scaling
                    # so we're left with another 2D array..
                    if np.square(distance[k,l]) >= np.square(i - origin[0]) + np.square(j - origin[1]):
                        probability[k,l] += occupancy[i, j]
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
            probs = integratedProbability(occ, radiiList[:,t,:])
            probabilityFile[:,t,:] = probs  # shape: (# scalings, # velocities)
            # save. note that this overwrites the file each time
            # structure is (scaling, times, velocities)
            np.save(saveFile, probabilityFile)

if __name__ == "__main__":
    # mem efficient versino of runQuadrantsData.py???
    #Check for files, set everything up
    # set up occ and get list of ts, then calculate radii
    # TODO: put in function so can do python3 memEfficientEvolve2DLattice arg1 arg2 arg3 ??
    def runDirichlet(L, tMax, alphas, velocities, saveFile):
        # setup
        alphas = np.array(alphas)
        occ = np.zeros((2 * L + 1, 2 * L + 1))
        occ[L, L] = 1
        ts = ev.getListOfTimes(1, tMax)  # array of times
        # TODO: fix velocity calc. to reflect the "want velocities kinda close to 1 but not quite at 1'
        # for velocities: want... 10^-3 to 4?
        #velocities = np.array([[]])  # needs to be (1, #ofVelocities) in shape?
        if velocities is None:
            velocities = np.array([np.geomspace(10**(-3),10,21)])  # the extra np.array([]) outside is to get the correct shape
        # get list of radii, scaling order goes linear, sqrt, tOnLogT, tOnSqrtLogT
        listOfRadii = np.array([calculateRadii(ts, velocities, linear), calculateRadii(ts,velocities,np.sqrt),
                                calculateRadii(ts, velocities, tOnLogT), calculateRadii(ts, velocities, tOnSqrtLogT)])

        # check if savefile exists already and is complete?
        if os.path.exists(saveFile):
            data = pd.read_csv(saveFile)
            max_time = max(data['Time'].values)
            if max_time == ts[-2]:
                print(f"File Finished", flush=True)
                sys.exit()
        # actually run and save data
        evolveAndMeasurePDF(ts, tMax, occ, listOfRadii, alphas, saveFile)

    # these should call sysargv now, or argparse
    # L =
    # tMax =
    # alphas =
    # velocities = np.array([velocitiy 1, velocity 2, velocity3])
    # saveFile =
    # initialize occ. & get list of t's

    # actually call runDirichlet here, inside the if __name__ == __main__

#
# # TODO: turn this into the above shit
# def measureDirichletPastBox(tMax, L, vs, alphas, barrierScale, saveFile):
#     """ memory efficient version of measureAtVsBox
#     :param tMax: int, maximum time
#     :param L: dist. from origin to edge of occ array
#     :param vs: np array, list of velocities at which barrier is moving
#     :param alphas: np array, list of 4 values of alpha parameter for Dirichlet dist.
#     :param barrierScale: str, either 't' 't**(1/2)' or something
#     :param saveFile: str, path to which data is written
#     """
#     #initialize occupancyi
#     occ = np.zeros((2 * L + 1, 2 * L + 1))
#     occ[L, L] = 1
#     ts = ev.getListOfTimes(1, tMax)
#     print(f"ts: {ts}")
#     # check if savefile exists already and is complete?
#     if os.path.exists(saveFile):
#         data = pd.read_csv(saveFile)
#         max_time = max(data['Time'].values)
#         if max_time == ts[-2]:
#             print(f"File Finished", flush=True)
#             sys.exit()
#     # set up writer and write header if save file doesn't exist
#     with open(saveFile, 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Time", *vs])
#         # generate data
#         for t, occ in evolve2DDirichlet(occ, tMax, alphas):
#             print(f"t in loop: {t}")
#             if t in ts:
#                 # get list of radii dependent on velocities
#                 RsScale = eval(barrierScale)
#                 Rs = list(np.array(vs * RsScale))
#                 # calculate probabilities & save them to file
#                 probs = [integratedProbability(occ, r) for r in Rs]
#                 writer.writerow([t, *probs])
#                 f.flush()
#         f.close()
