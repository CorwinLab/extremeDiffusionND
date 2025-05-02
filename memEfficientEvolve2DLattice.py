import numpy as np
import os
from time import time as wallTime  # start = wallTime() to avoid issues with using time as variable
from numba import njit
from randNumberGeneration import getRandomDistribution
import json
from datetime import date
import h5py
import sys
import npquad


@njit
def updateOccupancy(occupancy, time, func):
    """
	memory efficient version of executeMoves from evolve2DLattice
	:param occupancy: np array of size L
	:param time: the timestep at which the moves are being executed (int)
	:param func: function that returns random numbers

	Examples
	--------
	# Get speed of updateOccupancy
	import time

	func = getRandomDistribution('Dirichlet', [1, 1, 1, 1])

	L = 1000
	occ = np.zeros((2 * L + 1, 2 * L + 1))
	occ[L, L] = 1

	start = time.time()
	for t in range(500):
		occ = updateOccupancy(occ, t, func)
	print(time.time() - start)
	"""

    origin = occupancy.shape[0] // 2

    # TODO: at some point implement "find indx of farthest occupied site
    # and sweep over that square instead
    # this has an upper limit since vt grows linearly with t.
    # for short times, only loop over the part of the array we expect to be occupied
    if time < origin:
        startIdx = origin - time
        endIdx = origin + time
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
                biases = func()

                occupancy[i, j - 1] += occupancy[i, j] * biases[0]  # left
                occupancy[i + 1, j] += occupancy[i, j] * biases[1]  # down
                occupancy[i - 1, j] += occupancy[i, j] * biases[2]  # up
                occupancy[i, j + 1] += occupancy[i, j] * biases[3]  # right

                # set original site to 0
                occupancy[i, j] = 0

    return occupancy


def evolve2DDirichlet(occupancy, maxT, func, startT=1):
    """ generator object, memory efficient version of evolve2DLattice
	note that there's no absorbingBoundary because of the way i and j are indexed
	in updateOccupancy
	:param occupancy: np array, inital occupancy
	:param maxT: int, the final time to which system is evolved
	:param func: function that returns random numbers
	:param startT: optional int default 1, the time at which you want to start evolution
	"""
    for t in range(startT, maxT):
        occupancy = updateOccupancy(occupancy, t, func)
        yield t, occupancy


@njit
def integratedProbability(occupancy, distances, time):
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

    # TODO: at some point implement "find indx of farthest occupied site
    # and sweep over that square instead
    # this has an upper limit since vt grows linearly with t.
    # for short times, only loop over the part of the array we expect to be occupied
    if time < origin[0]:
        startIdx = origin[0] - time
        endIdx = origin[0] + time
    else:
        # start at 1 and go to shape-1 because otherwise at the boundary which we don't want
        # this is an effective way of implementing absorbing boundary conditions
        # note that the boundary is square and not circular
        startIdx = 1
        endIdx = occupancy.shape[0] - 1

    # iterate over the current occupancy
    for i in range(startIdx, endIdx):
        for j in range(startIdx, endIdx):
            # Move to next site if there's no probability
            if occupancy[i, j] == 0:
                continue

            # iterate over the the radii that are passsed in
            for k in range(distances.shape[0]):
                for l in range(distances.shape[1]):
                    # <= because we have we want (distToCenter >= radii) to get outside sphere
                    # and the line below has (radii <= dist)
                    if np.square(distances[k, l]) <= np.square(i - origin[0]) + np.square(j - origin[1]):
                        probability[k, l] += occupancy[i, j]
    return probability


def calculateRadii(times, velocity, scalingFunction):
    """
	get list of radii = v*(function of time) for barrier for given times; returns array of (# times, # velocities)
	Ex: radii = calculateRadii(np.array([1,5,10]),np.array([[0.01,0.1,0.5]]),tOnLogT)
	To get the original velocities, call radiiVT[0,:]
	"""

    funcVals = scalingFunction(times)
    funcVals = np.expand_dims(funcVals, 1)
    return velocity * funcVals


def linear(time):
    return time


def tOnSqrtLogT(time):
    with np.errstate(divide='ignore'):
        return time / np.sqrt(np.log(time))


def tOnLogT(time):
    with np.errstate(divide='ignore'):
        return time / np.log(time)


def constantRadius(time):
    # for a fixed radius, it just be 1 (and then it gets called as radii = v*constantRadius

    return np.full_like(time.astype(float), fill_value=1, dtype=np.float64)


def sqrt(time):
    return np.sqrt(time)


def getListOfTimes(maxT, startT=1, num=500):
    """
	Generate a list of times, with approx. 10 times per decade (via np.geomspace), out to time maxT
	:param maxT: the maximum time to which lattice is being evolved
	:param startT: initial time at which lattice evolution is started
	:param num: number of times you want
	:return: the list of times
	"""
    return np.unique(np.geomspace(startT, maxT, num=num).astype(int))


def evolveAndMeasurePDF(ts, startT, tMax, occupancy, func, saveFileName, tempFileName):
    """
	evolves occupancy lattice and makes probability lattice, through the generator loop

	ts: np array (ints) of times
	startT: int; the start time at which evolution is starting/continuing
	tMax: int; final time to which occupancy is evolved
	occupancy: 2d np array; either full of 0s with 1 at middle, or a loaded in state
	radiiList: 3d np array; lists the radii (floats) past which probability measurements are made
	alphas: np array, floats; array of alpha1=alpha2=alpha3=alpha4 for dirichlet distribution
	saveFile: h5 object; this is the file we are going to be saving data to
	"""
    startTime = wallTime()

    for t, occ in evolve2DDirichlet(occupancy, tMax, func, startT):
        if t in ts:
            # First need to pull out radii at current time we want
            idx = list(ts).index(t)
            radiiAtTimeT = []

            with h5py.File(saveFileName, 'r+') as saveFile:
                # Change because radii are now attached to the save file
                for regimeName in saveFile['regimes'].keys():
                    radii = saveFile['regimes'][regimeName].attrs['radii']
                    regimeRadiiAtTimeT = radii[idx, :]
                    radiiAtTimeT.append(regimeRadiiAtTimeT)

                # Shape of resulting array is (regimes, velocities)
                radiiAtTimeT = np.vstack(radiiAtTimeT)
                # TODO: check if min of integrated probs is also 1e-45... save separately
                probs = integratedProbability(occ, radiiAtTimeT, t)
                print(f"probs min (integratedProb: {np.nanmin(probs[probs!=0])}")
                print(f"dtype probs: {probs.dtype}")

                # Now save data to file
                for count, regimeName in enumerate(saveFile['regimes'].keys()):
                    saveFile['regimes'][regimeName][idx, :] = probs[count, :]

                # For ease, we will save the occupancy every time we
                # write data to the file
                saveFile.attrs['currentOccupancyTime'] = t
                # TODO: put this bit into a helper function?
                # saveFile['currentOccupancy'][:] = occ
                print(f"creating temp file {tempFileName} at t = {t}")
                with h5py.File(tempFileName, "a") as temp:
                    temp.create_dataset('currentOccupancy', data=occ, compression='gzip', dtype=np.float64)
                print(f"copying temp to main")
                saveFile['currentOccupancy'][:] = h5py.File(tempFileName, "r")['currentOccupancy']
                # saveFile['currentOccupancy'][:] = occ
                # # TEMPORARY SYS.EXIT() FOR DEBUGGING PURPOSES
                # sys.exit()
                print(f"deleting temp {tempFileName}")
                os.remove(tempFileName)
                startTime = wallTime()

        hours = 3
        seconds = hours * 3600
        # Save at final time and if haven't saved for XX hours
        # Because it might take longer than 3 hours to go between
        # save times
        if (wallTime() - startTime >= seconds) or (t == tMax - 1):
            # Save current time and occupancy to make restartable
            with h5py.File(saveFileName, 'r+') as saveFile:
                saveFile.attrs['currentOccupancyTime'] = t
                print(f"creating temp file {tempFileName} at t = {t}")
                with h5py.File(tempFileName,"a") as temp:
                    temp.create_dataset('currentOccupancy', data=occ, compression='gzip',dtype=np.float64)
                    print(f"copying temp to main")
                    saveFile['currentOccupancy'][:] = h5py.File(tempFileName, "r")['currentOccupancy'][:]
                # saveFile['currentOccupancy'][:] = occ
                print(f"deleting temp {tempFileName}")
                os.remove(tempFileName)
            # Reset the timer
            startTime = wallTime()


def runSystem(L, ts, velocities, distName, params, directory, systID):
    """
	memory efficient eversion of runQuadrantsData.py; evolves with a bajillion for loops
	instead of vectorization, to avoid making copies of the array, to save memory.

	L: int, distance from origin to edge of array
	ts: numpy array, times to save at 
	velocities: numpy array, velocities to measure at 
	distname: string, name of distribution ('Dirichlet', 'Delta', 'SymmetricDirichlet')
	params: string, parameters for the corresponding distribution
	saveFile: str, base directory to which data is saved
	systID: int, number which identifies system
	"""

    # setup random distribution
    func = getRandomDistribution(distName, params)
    ts = np.array(ts)
    velocities = np.array(velocities)
    tMax = max(ts) + 1  # should be 10,000
    # without the +1, this returns tMax (from bash file)-1
    # and then it gets passed into evolveAndMeasure(..,tMax,..) which then uses it
    # in "for t in range(startT, tMax))" but that will run to 1 less than the parameter given
    # hence 1998
    saveFileName = os.path.join(directory, f"{str(systID)}.h5")
    tempFileName = os.path.join(directory,"temp"+f"{str(systID)}.h5")

    with h5py.File(saveFileName, 'a') as saveFile:
        # Define the regimes we want to study
        regimes = [linear, np.sqrt, tOnSqrtLogT]
        # Check if "regimes" group has been made and create otherwise
        if 'regimes' not in saveFile.keys():
            saveFile.create_group("regimes")

            for regime in regimes:
                saveFile['regimes'].create_dataset(regime.__name__, shape=(len(ts), len(velocities)),dtype=np.float64)
                saveFile['regimes'][regime.__name__].attrs['radii'] = calculateRadii(ts, velocities, regime)

        # Load save if occupancy is already saved
        # Eric says the following should be a function.
        if ('currentOccupancyTime' in saveFile.attrs.keys()) and ('currentOccupancy' in saveFile.keys()):
            # TODO: atomic operations here, check for temp file
            if os.path.isfile(tempFileName):
                # if the temp file exists, then the old file (systID.h5)
                # has not been re-written to, yet. but there's probably something
                # wrong with the temp file. therefore we want to reload the old file occupancy
                # and set the time to be that of the old file's currentOccupancyTime?
                print("temp file exists, restoring old file")
                mostRecentTime = saveFile.attrs['currentOccupancyTime']
                occ = saveFile['currentOccupancy'][:]  # array vs hdf5 dataset
                # delete the bad temp file...
                print('deleting bad temp file')
                os.remove(tempFileName)
            else:
                print('restoring most recent save')
                mostRecentTime = saveFile.attrs['currentOccupancyTime']
                occ = saveFile['currentOccupancy'][:]
        else:
            # Otherwise, initialize as normal
            occ = np.zeros((2 * L + 1, 2 * L + 1))
            occ[L, L] = 1
            mostRecentTime = 1
            saveFile.create_dataset('currentOccupancy', data=occ, compression='gzip',dtype=np.float64)
            saveFile.attrs['currentOccupancyTime'] = mostRecentTime

    # actually run and save data
    evolveAndMeasurePDF(ts, mostRecentTime, tMax, occ, func, saveFileName, tempFileName)

    # To save space we delete the occupancy when done
    # with h5py.File(saveFileName, 'r+') as saveFile:
    #     del saveFile['currentOccupancy']


def getExpVarX(distName, params):
    '''
	Calculates Var_nu[E^xi[X]] numerically using jacob's special vector definition
	Examples
	--------
	alpha = 0.1
	var = getExpVarX('Dirichlet', [alpha] * 4)
	print(var, 1 / (1 + 4 * float(alpha)))
	'''

    func = getRandomDistribution(distName, params)
    ExpX = 0
    xvals = np.array([-1, -1, 1, 1])
    num_samples = 100000
    for _ in range(num_samples):
        rand_vals = func()
        ExpX += np.sum(
            xvals * rand_vals) ** 2  # 16 terms because 4 terms in the (xvals*randvals) quantity, and then you sum and square them?
    ExpX /= num_samples

    return ExpX


#TODO (sometime-ish): rewrite above w/ dot product scheme and verify its off by a factor of 1/2 or 2 or something
def getExpVarXDotProduct(distName, params):
    """
    Calculates Var_nu[E^xi[X]] numerically using vector variance (dot product) definition
	Examples
	--------
	alpha = 0.1
	var = getExpVarX('Dirichlet', [alpha] * 4)
	print(var, 1 / (1 + 4 * float(alpha)))
    """
    func = getRandomDistribution(distName, params)
    num_samples = 100000
    ExpX = 0
    # nvecs instead of xvals because we need to preserve x and y orthogonality
    # nvecs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    for _ in range(num_samples):
        # rand_vals = np.array(func()).reshape(-1, 1)
        # temp = rand_vals * nvecs  # should be (4,2) in shape
        # # calc dot products (this replaces the sum squared)
    	# # xi(epsilon nhat) xi(epsilon' nhat') epsilon nhat' dot epsilon' nhat'
        # dotProduct = 0
        # for i in range(4):
        #     for j in range(4):
        #        dotProduct += np.dot(temp[i],temp[j])

        # use eqn. 9 from my writeup
        rand_vals = np.array(func())  # left (-x), down(-y), up(y), right(x)
        dotProduct = (rand_vals[3]**2 + rand_vals[0]**2 + rand_vals[2]**2 + rand_vals[1]**2
                      -2*rand_vals[3]*rand_vals[0] - 2*rand_vals[2]*rand_vals[1])
        ExpX += dotProduct
    ExpX /= num_samples
    return ExpX


def diamondCornerVariance(t,distName='Dirichlet',params=''):
    """ calculate var[lnP] of rwre being at the 4 corners
    process (equiv of setting v=1 for vt regime):
    take product of t dirichlet numbers, 4 times independently
    repeat a ton of times
    then take log of all of them and calculate variance
    note because log(product) is sum(log) we can add instead?
    for now only going to do this for dirichlet as a check.
    which means lambda_ext will be for dirichlet with alpha=0.1
    """
    if distName == 'Dirichlet':
        # for now default to alpha=0.1
        params = np.array([0.1]*4)
    # func = getRandomDistribution(distName, params)
    probs = []
    logProbs = []
    num_samples = 1000
    for _ in range(num_samples):
        #rand_vals = np.array([func() for i in range(t)],dtype=np.quad)
        probSum = 0
        for direction in range(4):
            rand_vals = np.array(np.random.dirichlet(params,size=t),dtype=np.quad)
            product = np.prod(rand_vals[:,0])
            probSum += product
        probs.append(probSum)
        logP = np.log(probSum)
        logProbs.append(logP)
    var = np.var(logProbs)
    return np.array(probs,dtype=np.float64), np.array(logProbs,dtype=np.float64), np.float64(var)

def saveVars(vars, save_file):
    """
	Save experiment variables to a file along with date it was ran
	"""
    for key, item in vars.items():
        if isinstance(item, np.ndarray):
            vars[key] = item.tolist()

    with open(save_file, "w+") as file:
        json.dump(vars, file)


if __name__ == "__main__":
    # Test Code
    # L, tMax, distName, params, directory, systID = 5000, 10000, 'Dirichlet', '1,1,1,1', './', 0

    L = int(sys.argv[1])
    tMax = int(sys.argv[2])
    distName = sys.argv[3]
    params = sys.argv[4]
    directory = sys.argv[5]
    systID = int(sys.argv[6])

    # Need to parse params into an array unless it is an empty string
    if params == 'None':
        params = ''
    else:
        params = params.split(",")
        params = np.array(params).astype(float)
        print(f"params: {params}")

    ts = getListOfTimes(tMax - 1, 1)  # 1 to 10,000-1
    velocities = np.geomspace(10 ** (-5), 10, 21)

    vars = {'L': L,
            'ts': ts,
            'velocities': velocities,
            'distName': distName,
            'params': params,
            'directory': directory,
            'systID': systID}
    print(f"vars: {vars}")
    os.makedirs(directory, exist_ok=True)  # without this, gets mad that directory might not fully exist yet
    vars_file = os.path.join(directory, "variables.json")
    print(f"vars_file is {vars_file}")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    # Only save the variables file if on the first system
    if systID == 0:
        print(f"systID is {systID}")
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")

    start = wallTime()
    runSystem(**vars)
    print(wallTime() - start, flush=True)
