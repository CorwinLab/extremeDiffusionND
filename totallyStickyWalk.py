import numpy as np
import differenceRandomWalk as drw
import os
from tqdm import tqdm
from directedPolymer import logSumExp
from numba import njit
from scipy.stats import skew


# this module is to run the totally sticky random walk statistics
# so that we can plot log[varP] at t=1000 for different radii and times as a func
# of velocity, g(2arctanh(v), alpha), and beta

@njit
def stickyXVal(dList, tMax):
    """
    returns the final x value of a totally sticky random walk in d=2 lattice
    dList: np array, list of nearest neighbors
    tMax: int, longest time for which walk should evolve
    """
    finalX = 0  # the final x position is just the sum of each individual jump's x direction
    for t in range(tMax):
        rand_jump = np.random.randint(4)  # can't do size=tMax in numba
        finalX += dList[rand_jump, 0]  # grab the direction indicated, then add only the x val
    return finalX

@njit
def stickyXVals(dList, tMax, numSystems):
    """
    returns n=numSystems totally sticky random walk final x positions
    dList: np array, list of  nearest neighbors
    tMax: int, longest time for which walk should evolve
    numSystems: int, total number of systems
    """
    stickyXVals = np.full(numSystems, np.nan)
    for i in range(numSystems):
        stickyXVals[i] = stickyXVal(dList, tMax)
    return stickyXVals


def getStats(finalXVals, r):
    """ returns the probability that lastXVals are bigger than or equal to r or not """
    condition = (finalXVals >= r).astype(int)

    return np.mean(condition), np.var(condition), skew(condition)

def runLineProbs(velocities, tMax,
                numSystems=30000,
                saveDir="/home/fransces/Documents/code/extremeDiffusionND/pastLine/totallySticky"):
    """
    records the logProbs for a list of given radii (given velocities) at time t=tMax of the sticky random walks
    """
    os.makedirs(saveDir, exist_ok=True)
    velocities = np.array(velocities)
    dList = drw.makeDirectionList()  # get nearest neighbors
    # prep measurement distances for each regime
    sqrtRadii = (velocities * np.sqrt(tMax))
    criticalRadii = (velocities * tMax/np.sqrt(np.log(tMax)))
    linearRadii = (velocities * tMax)
    # this should be a # oft by 3*# ofvelocities 2d array, e.g. a (150,)
    radiiArray = np.hstack((sqrtRadii, criticalRadii, linearRadii))
    probArray = np.full((3,radiiArray.shape[0]), np.nan)  # (# stats by 150
    # Compute the set of finalXVals only once, and then make measurements on them
    finalXVals = stickyXVals(dList, tMax, numSystems)
    for i, r in enumerate(radiiArray):
        probArray[:,i] = getStats(finalXVals, r)
    np.save(os.path.join(saveDir, "radiiProbArray.npy"), np.vstack([radiiArray, probArray]))

    return radiiArray, probArray


