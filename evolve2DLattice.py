import numpy as np
import os
from scipy.ndimage import morphology as m
# import time
import csv
import npquad


def doubleArray(array, arraytype, fillValue=0):
    """
    Takes an existing numpy array and doubles it, keeping existing vals at center
    :param array: the array you want to double, assumes square and odd size
    :param arraytype: tell the dtype of the array (int, float, etc.)
    :param fillValue: what do you want to fill the new entries with, automatically set to 0
    :return newArray (the inputted array but doubled)
    """
    length = (array.shape[0]) // 2
    if length == 0:
        newLength = 1
    else:
        newLength = 2 * length
    newArray = np.full((2 * newLength + 1, 2 * newLength + 1), fillValue, dtype=arraytype)
    newArray[newLength - length:newLength + length + 1, newLength - length:newLength + length + 1] = array
    return newArray


def getRandVals(distribution, rng, shape, params):
    """
    Get random values across the lattice with specified shape according to the specified distribution
    """
    if distribution == 'uniform':
        biases = rng.dirichlet([1] * 4, shape)
    elif distribution == 'SSRW':
        biases = np.full([shape, 4], 1 / 4)
    elif distribution == 'dirichlet':
        biases = rng.dirichlet([params] * 4, shape)
    return biases


def executeMoves(occupancy, i, j, rng, distribution, PDF, params=None):
    """
    Evolves 2Dlattice according to a distribution, with the option to evolve agents or the PDF
    :param occupancy: the array you are working in or the initial occupancy
    :param i, j = sites with agents in them/sites that are occupied
    :param rng: numpy random number generator (should be np.random.default_rng() passed in)
    :param distribution: string specifying distribution you are using to generate biases
    :param PDF: boolean; if true then multiplies biases; if false then draws multinomial\
    :param params: the parameters of the distribution
    :return occupancy: the new occupancy after moving everything
    """
    # Generate biases for each site
    biases = getRandVals(distribution, rng, i.shape[0], params)

    # On newer numpy we can vectorize to compute the moves
    if PDF:  # if doing PDF then multiply by biases
        moves = occupancy[i, j].reshape(-1, 1) * biases  # reshape -1 takes the shape of occupancy
    else:
        moves = rng.multinomial(occupancy[i, j], biases)
    # Note that we can use the same array because we're doing checkerboard moves
    # If we want to use a more general jump kernel we need to use a new (empty) copy of the space
    occupancy[i, j - 1] += moves[:, 0]  # left
    occupancy[i + 1, j] += moves[:, 1]  # down
    occupancy[i, j + 1] += moves[:, 2]  # right
    occupancy[i - 1, j] += moves[:, 3]  # up
    occupancy[i, j] = 0  # Remove everything from the original site, as it's moved to new sites
    return occupancy


def changeArraySize(array, size, fillval):
    """
    Takes an existing numpy array and expands it to the size you want to change it to with a fill value
    :param array: the array you want to change the size of
    :param size: the new size (as in LxL)
    :param fillval: what you want to fill the new entries with
    :return: newArray: the old array but expanded to the specified size, with the new entried filled
    """
    length = (array.shape[0]) // 2
    newsize = size // 2
    if length < size:
        newArray = np.full((2 * newsize + 1, 2 * newsize + 1), fillval, dtype=int)
        newArray[newsize - length:newsize + length + 1, newsize - length:newsize + length + 1] = array
    else:
        newArray = array
    return newArray


# main functions & generators + wrappers
def evolve2DLattice(occupancy, maxT, distribution, params, PDF, occtype, startT=1,
                    rng=np.random.default_rng(), boundary=True):
    """
    generator; evolves agents in a 2D lattice out to some time maxT with dynamic scaling.
    :param occupancy: either number (NParticles) or occupancy array
    :param maxT: timestep you want to go out to
    :param distribution: string, specify the distribution of biases
    :param params: parameters of distribution
    :param PDF: boolean; if true then multiplies biases; if false then evolves agents
    :param occtype: the dtype of the occupancy array (int for agents, float for pdf)
    :param startT: optional; time you want to start at; default 1
    :param rng: the numpy random number generator obj(default np.random.default_rng() )
    :param boundary: (numpy array) boundary conditions should be same size as occupancy

    :return t: yield tArrival array
    :return occupancy: yield occupancy array
    """
    for t in range(startT, maxT):
        # Find the occupied sites
        i, j = np.where((occupancy != 0) & boundary)
        # If the occupied sites are at the limits (i.e if min(i,j) = 0 or max(i,j) = size)
        # then we need to enlarge occupancy and create a new array. Don't resize if 
        # boundary is specified.
        if ((np.min([i, j]) <= 0)
                or (np.max([i, j]) >= np.min(occupancy.shape) - 1)
                and not (isinstance(boundary, np.ndarray))):
            occupancy = doubleArray(occupancy, occtype)
            # These next two lines are a waste; we could just do index translation
            sites = (occupancy != 0)
            i, j = np.where(sites & boundary)
        occupancy = executeMoves(occupancy, i, j, rng, distribution, PDF, params)
        yield t, occupancy


def generateFirstArrivalTime(occupancy, maxT, distribution, params, PDF, startT=1):
    """
    Evolves a 2DLattice with dynamic scaling.
    :param occupancy: initial occupancy, can be a number (NParticles) or an existing array
    :param maxT: timestep you want to go out to (integer)
    :param distribution: string; specify distribution from which biases pulled
    :param params: the parameters of the specified distribution
    :param PDF: boolean; if true, evolves PDF, if false, uses multinomial to evolve agents
    :param startT: optional arg, starts at 1
    :return occ: the final evolved occupancy array
    :return tArrival: the array with the time of first arrival for every site in the occupancy array
    """
    notYetArrived = -1
    if PDF and occupancy != 1:
        print("Warning: You are trying to evolve a PDF with N!= 1, so it won't be normalized.")
    # deal with the dtype messiness depending on if you want agents or a PDF
    if PDF:
        occtype = float
    else:
        occtype = int
    # if given a scalar (ie NParticles or something), initializes array
    if distribution == 'dirichlet':
        params = float(params)
    else:
        params = None
    if np.isscalar(occupancy):
        occupancy = np.array([[occupancy]], dtype=occtype)
    # initialize the array to throw in the time of first arrivals
    tArrival = np.copy(occupancy)
    tArrival[:] = notYetArrived
    # use evolve2DLattice to evolve; record tArrivals and throw them into array as lattice evolves
    for t, occ in evolve2DLattice(occupancy, maxT, distribution, params, PDF, occtype):
        if tArrival.shape[0] != occ.shape[0]:
            # Note: this is fragile, we assume that doubling tArrival will always work
            tArrival = doubleArray(tArrival, arraytype=int, fillValue=notYetArrived)
        tArrival[(occ > 0) & (tArrival == notYetArrived)] = t
        # if t % 1000 == 0:
        #     np.savez_compressed(f"testData3/t{t}.npz", tArrival=tArrival, occ=occ)
    return tArrival, occ


# wrapper for evolve2DLattice
def run2dAgent(occupancy, maxT, distribution, params, PDF, occtype):
    for t, occ in evolve2DLattice(occupancy, maxT, distribution, params, PDF, occtype):
        pass
    return t, occ


# data analysis functions

# take a path with files from runFirstArrivals and get mean(tArrival), var(tArrival), mask of tArrivals
# this calculates mean and var by hand when loading in every tArrival will take too much memory
def getTArrivalMeanAndVar(path):
    """
    Takes a directory filled with tArrival arrays and finds the mean and variance of tArrivals
    :param path: the path of the directory
    :return: finalMom1: the first moment (mean) of tArrivals
    :return finalMom2 - finalMom1**2: the variance of the tArrivals
    :return goodData: the mask, so you only look at the stuff where every agent has gotten to
    """
    filelist = sorted(os.listdir(path))
    notYetArrived = -1
    # initialize the moments & mask
    # tArrMom1, tArrMom2, goodData = None, None, None
    tArrMom1 = None  # moment 1 is just t
    tArrMom2 = None  # moment 2 is t**2
    goodData = None  # the mask
    # go through each file and pull out the tArrival array
    for file in filelist:
        # this should return ONE tArrival array
        tArrival = np.load(f'{path}/{file}')['tArrival']
        # if you are on the first file, make the moment arrays using the first file
        if tArrMom1 is None:
            tArrMom1 = tArrival
            tArrMom2 = tArrival ** 2
            goodData = (tArrival != -1)
        # if you are somewhere in the middle of the list, first check that the array sizes will be the same
        # or alternatively make them the same
        else:
            if tArrMom1.shape[0] < tArrival.shape[0]:
                # if not the same size, change the moment arrays to be the same size
                # as the incoming tArrival array
                tArrMom1 = changeArraySize(tArrMom1, tArrival.shape[0], notYetArrived)
                tArrMom2 = changeArraySize(tArrMom2, tArrival.shape[0], notYetArrived)
                goodData = changeArraySize(goodData, tArrival.shape[0], notYetArrived)
            if tArrMom1.shape[0] > tArrival.shape[0]:
                # if not the same size, change the moment arrays to be the same size
                # as the incoming tArrival array
                tArrival = changeArraySize(tArrival, tArrMom1.shape[0], notYetArrived)
            # now cumulatively add the moments
            # need to deal with the -1 thing... my variance has a range of -20 to 0
            goodData *= (tArrival != notYetArrived)
            tArrMom1 += tArrival * goodData
            tArrMom2 += (tArrival * goodData) ** 2
    finalMom1 = tArrMom1 / len(filelist)
    finalMom2 = tArrMom2 / len(filelist)
    # Return the mean and the variance, and the mask
    return finalMom1, finalMom2 - finalMom1 ** 2, goodData


def cartToPolar(i, j):
    """
    Can take indices (i,j) and turn them into polar coords. r, theta
    Note: indices need to be already shifted so origin is at center appropriately
    :param i: the (down? vertical?) index in cartesian coordinates
    :param j: the (horizontal? across?) index in cartesian coordinates
    :return r, theta: the polar coords of indices i,j
    """
    r = np.sqrt(i ** 2 + j ** 2)
    theta = np.arctan2(j, i)
    return r, theta


# roughness statistics functions
def getPerimeterAreaTau(tArrivalArray, tau):
    """
    Goes inside getRoughness. Finds Roughness parameters for a single array at specific time tau
    :param tArrivalArray: array of tArrivals
    :param tau: the time at which you want to measure roughness (int)
    :return: perimeter: the number of pixels on the boundary at time tau
    :return: area: the number of pixels within the area of the boundary at tau
    :return: tau: the time at which you've specified to measure roughness
    :return: boundaryDist: the <(distance to origin of boundary)> (to calculate moments) at time tau
    :return: boundaryDist2: the <(dist to origin of boundary)**2> (to calculate moments) at time tau
    """
    notYetArrived = -1
    mask = ((tArrivalArray <= tau) & (tArrivalArray > notYetArrived))
    boundary = (mask ^ m.binary_erosion(mask))
    # get distance to origin of boundary points
    i, j = np.where(boundary)
    L = tArrivalArray.shape[0] // 2
    i, j = i - L, j - L
    r, theta = cartToPolar(i, j)
    boundaryDist = np.mean(r)
    boundaryDist2 = np.mean(np.square(r))
    perimeter = np.sum(boundary)
    area = np.sum(mask)
    roughness = perimeter/np.sqrt(area)
    return perimeter, area, roughness, boundaryDist, boundaryDist2, tau


# 9 April getRoughnessNew --> getRoughness
# avgdist = <r> and avgdist2 = <r^2>
def getRoughness(tArrivalArray):
    """
    Calculate the surface roughness of tArrivals as a function of time by
    (#boundary pixels)/(total pixels reached)^1/2
    If perfectly smooth, min roughness is like 2*(pi)^1/2.
    Takes one tArrival array and roughness stats. as function of tau
    :param tArrivalArray: the tArrival array, going out to max time tau
    :return stats: numpy array of shape (tau, 5);
        stats[:,0] returns perimeter, [:,1] area, [:,2]  roughness
        [:,3]  avg. dist to boundary, [:,4] returns (avg dist to boundary)^2,
        [:5] returns tau
    """
    # initialize stats with the tau = 1 (since startT = 1 in generateFirstArrivals)
    stats = getPerimeterAreaTau(tArrivalArray, 1)  # p,a,r,d,d2,t
    # add the rest of the stats using vstack
    for i in range(2, np.max(tArrivalArray)):
        tempstats = getPerimeterAreaTau(tArrivalArray, i)
        stats = np.vstack((stats, tempstats))
    return stats


# 9 April getRoughnessMeanVarNew --> getRoughnessMeanVar
def getRoughnessMeanVar(path):
    """
    Take directory of tArrivals statistics (from np.save (getRoughness())), and calculates
    mean, 2nd moment, + var of each stat. Uses npquad for precision
    :param path: directory name of tArrival stats (from getRoughness saved as npy)
    :return: mean: the list of means of each stat (as function of tau)
    :return moment2: the list of second moments of each stat (as function of tau)
    :return: var: the list of variances of each stat (as function of tau)
    """
    filelist = sorted(os.listdir(path))
    # initialize array
    firstStats = np.load(f'{path}/{filelist[0]}').astype(np.quad)
    mean = firstStats
    moment2 = firstStats ** 2
    # cumulatively calculate mean, second moment
    for file in filelist[1:]:
        tempStats = np.load(f'{path}/{file}').astype(np.quad)
        # cumulatively add to the mean and second moment
        mean += tempStats
        moment2 += tempStats**2
    mean = mean / len(filelist)  # normalize
    moment2 = moment2 / len(filelist)
    var = moment2 - mean**2
    return mean, moment2, var


def getIndecesInsideSphere(occ, r):
    x = np.arange(-(occ.shape[0] // 2), occ.shape[0] // 2 + 1)
    xx, yy = np.meshgrid(x, x)

    dist_from_center = np.sqrt(xx ** 2 + yy ** 2)
    indeces = np.where(dist_from_center < r)
    return indeces


def getLineIndeces(occ, r):
    x = np.arange(-(occ.shape[0] // 2), occ.shape[0] // 2 + 1)
    xx, yy = np.meshgrid(x, x)

    indeces = np.where(xx >= r)
    return indeces


def measureOnSphere(tMax, L, R, Rs, distribution, params, sphereSaveFile, lineSaveFile):
    '''
    Parameters
    ----------
    L : int 
        Radius of size of box

    tMax : int 
        Maximum time to iterate to

    R : float
        Radius of circle for circular boundary conditions
    
    Example
    -------
    tMax = 100
    L = 250
    R = L-1
    Rs = [5, 10]
    savefile = 'PDF.txt'
    linefile = 'Line.txt'
    distribution = 'dirichlet'
    params = 1/10
    measureOnSphere(tMax, L, R, Rs, distribution, params, savefile, linefile)
    '''

    Rs.append(R)
    f = open(sphereSaveFile, 'a')
    writer = csv.writer(f)
    writer.writerow(["Time", *Rs])

    f_line = open(lineSaveFile, 'a')
    writer_line = csv.writer(f_line)
    writer_line.writerow(['Time', *Rs])

    # Create occupancy array
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1

    x = np.arange(-L, L + 1)
    xx, yy = np.meshgrid(x, x)
    dist_to_center = np.sqrt(xx ** 2 + yy ** 2)
    boundary = dist_to_center <= R

    indeces = [getIndecesInsideSphere(occ, r) for r in Rs]
    line_indeces = [getLineIndeces(occ, r) for r in Rs]
    ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
    # Need to make sure occ doesn't change size
    for t, occ in evolve2DLattice(occ, tMax, distribution, True, float, params=params, boundary=boundary):
        # Get probabilities inside sphere
        if t in ts:
            probs = [1 - np.sum(occ[idx]) for idx in indeces]
            writer.writerow([t, *probs])
            f.flush()

            # Get probabilities outside line
            probs = [np.sum(occ[idx]) for idx in line_indeces]
            writer_line.writerow([t, *probs])
            f_line.flush()

    f_line.close()
    f.close()


def measureAtVsOnSphere(tMax, L, R, vs , distribution, params, sphereSaveFile, lineSaveFile):
    '''
    Parameters
    ----------
    L : int 
        Radius of size of box

    tMax : int 
        Maximum time to iterate to

    R : float
        Radius of circle for circular boundary conditions
    
    Example
    -------
    tMax = 100
    L = 250
    R = L-1
    Rs = [5, 10]
    savefile = 'PDF.txt'
    linefile = 'Line.txt'
    distribution = 'dirichlet'
    params = 1/10
    measureOnSphere(tMax, L, R, Rs, distribution, params, savefile, linefile)
    '''
    
    f = open(sphereSaveFile, 'a')
    writer = csv.writer(f)
    writer.writerow(["Time", *vs])

    f_line = open(lineSaveFile, 'a')
    writer_line = csv.writer(f_line)
    writer_line.writerow(['Time', *vs])

    # Create occupancy array
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    
    x = np.arange(-L, L+1)
    xx, yy = np.meshgrid(x, x)
    dist_to_center = np.sqrt(xx ** 2 + yy ** 2)
    boundary = dist_to_center <= R

    ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
    # Need to make sure occ doesn't change size
    for t, occ in evolve2DLattice(occ, tMax, distribution, True, float, params=params, boundary=boundary):
        # Get probabilities inside sphere
        if t in ts: 
            Rs = list(np.array(vs * t**(3/4)).astype(int))
            
            indeces = [getIndecesInsideSphere(occ, r) for r in Rs]
            line_indeces = [getLineIndeces(occ, r) for r in Rs]

            probs = [1-np.sum(occ[idx]) for idx in indeces]
            writer.writerow([t, *probs])
            f.flush()

            # Get probabilities outside line
            probs = [np.sum(occ[idx]) for idx in line_indeces]
            writer_line.writerow([t, *probs])
            f_line.flush()

    f_line.close()
    f.close()

def measureRegimes(tMax, L, R, alpha, distribution, params, sphereSaveFile, lineSaveFile):
    '''
    Parameters
    ----------
    L : int 
        Radius of size of box

    tMax : int 
        Maximum time to iterate to

    R : float
        Radius of circle for circular boundary conditions
    
    Example
    -------
    tMax = 100
    L = 250
    R = L-1
    Rs = [5, 10]
    savefile = 'PDF.txt'
    linefile = 'Line.txt'
    distribution = 'dirichlet'
    params = 1/10
    measureOnSphere(tMax, L, R, Rs, distribution, params, savefile, linefile)
    '''
    
    f = open(sphereSaveFile, 'a')
    writer = csv.writer(f)
    writer.writerow(["Time", *alpha])

    f_line = open(lineSaveFile, 'a')
    writer_line = csv.writer(f_line)
    writer_line.writerow(['Time', *alpha])

    # Create occupancy array
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    
    x = np.arange(-L, L+1)
    xx, yy = np.meshgrid(x, x)
    dist_to_center = np.sqrt(xx ** 2 + yy ** 2)
    boundary = dist_to_center <= R

    ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
    # Need to make sure occ doesn't change size
    for t, occ in evolve2DLattice(occ, tMax, distribution, True, float, params=params, boundary=boundary):
        # Get probabilities inside sphere
        if t in ts: 
            Rs = list(np.array(1/2 * t**(np.array(alpha))).astype(int))
            
            indeces = [getIndecesInsideSphere(occ, r) for r in Rs]
            line_indeces = [getLineIndeces(occ, r) for r in Rs]

            probs = [1-np.sum(occ[idx]) for idx in indeces]
            writer.writerow([t, *probs])
            f.flush()

            # Get probabilities outside line
            probs = [np.sum(occ[idx]) for idx in line_indeces]
            writer_line.writerow([t, *probs])
            f_line.flush()

    f_line.close()
    f.close()







# OLD FUNCTIONS
# can maybe delete this now
# wrapper function for generateFirstArrivalTimeAgent, saves to a path; OLD? ISH?
# def runFirstArrivals(occupancy, MaxT, distribution, PDF, iterations, directoryName, params=None):
#     """
#     runs iterations of first arrival arrays, saves to a folder thats specified or created
#     Effectively does the same thing as runCopiesOfDataAndAnalysis bash script but in python
#
#     :param occupancy: existing occupancy array or a number(ie num. of particles)
#     :param MaxT: integer; number of timestseps you want to go out to
#     :param distribution: specify distribution from which biases are pulled
#     :param PDF: boolean; if true then multiplies biases; if false then draws multinomial
#     :param iterations: integer, number of iterations you want to run
#     :param directoryName: string, the path  name you want to create to save tArrivals to
#     :param params: parameters of the specified distribution
#     """
#     path = f"{directoryName}"
#     statsPath = f"{directoryName}" + "Statistics"
#     if PDF:  # to better label directories
#         path = path + "PDF"
#         statsPath = path + "Statistics"
#     # create a folder to throw all runs into, or check that it exists
#     if not os.path.exists(path):
#         os.mkdir(path)
#         os.mkdir(statsPath)
#         # os.chdir(path)
#         print(f"{directoryName} and {statsPath} have been created.")
#     else:
#         # os.chdir(f"{directoryName}")
#         print("folder exists")
#
#     # run your iterations and save each tArrival and occ array into the folder created/specified
#     for i in range(iterations):
#         tArrival, occ = generateFirstArrivalTime(occupancy, MaxT, distribution, PDF, params)
#         np.savez_compressed(f"{path}/{i}.npz", tArrival=tArrival, occ=occ)
#
#         if not PDF:  # if Agent (ie not PDF) then save stats; otherwise don't save them (for testing)
#             perimeter, area, time, roughness, avgDist, avgDist2 = getRoughness(tArrival)
#             # and save the analysis quantities to statspath/systid.npz
#             np.savez_compressed(f"{statsPath}/{i}.npz", perimeter=perimeter, area=area, time=time, roughness=roughness,
#                                 avgBoundaryDist=avgDist, avgBoundaryDist2=avgDist2)
#         print(i)

# def getRoughnessOld(tArrivalArray):
#     """
#     Takes one tArrival array and returns arrays of perimeter, area, time, and roughness (as function of tau)
#     :param tArrivalArray: numpy array generated from generateFirstArrivalTimeAgent
#     :return: perimeter: the array of number of pixels on the boundary
#     :return: area: the array of number of pixels within the area of the boundary
#     :return: tau: the array of time at which you've specified to measure roughness
#     :return: boundaryR: array of mean distance to origin of the boundary pixels (to calculate moments)
#     :return: boundaryR2: array of mean of (distance)^2 to origin of boundary pixels (to calculate moments)
#     """
#     # old way
#     perimeter = []
#     area = []
#     time = []
#     avgDist = []
#     avgDist2 = []
#     #for each file, pull out their perimeter, area, and time and append
#     #note that I think this assumes that the diffused particles aren't touching the edge
#     for i in range(1, np.max(tArrivalArray)):
#         # save as np. array instead?
#         p, a, r, d, d2, t = getPerimeterAreaTau(tArrivalArray, i) #p, a, d, d2, t
#         perimeter.append(p)
#         area.append(a)
#         time.append(t)
#         avgDist.append(d)
#         avgDist2.append(d2)
#     roughness = perimeter / np.sqrt(area)
#     return np.array(perimeter), np.array(area), np.array(roughness), np.array(avgDist), np.array(
#         avgDist2), np.array(time)
#
# def getRoughnessMeanVarOld(path):
#     """
#     Take directory of tArrivals, returns list of perimeters, areas, and also
#     returns mean(roughness) and var(roughness)?
#     :param path: directory name of statistics of tArrivals
#     :return: np array of perimeters, areas, and time, and mean(roughness), var(roughness)
#     """
#     filelist = sorted(os.listdir(path))
#     # initialize array of Roughness
#     PerimeterList = []
#     AreaList = []
#     TimeList = []
#     RoughnessList = []
#     dist = []
#     dist2 = []
#     for file in filelist:
#         tArrival = np.load(f'{path}/{file}')['tArrival']
#         tempP, tempA, tempR, tempD, tempD2, tempT = getRoughnessOld(tArrival) #p, a, r, d, d2, t
#         # for each file append the arrays of p,a,t to the list
#         PerimeterList.append(tempP)
#         AreaList.append(tempA)
#         TimeList.append(tempT)
#         RoughnessList.append(tempR)
#         dist.append(tempD)
#         dist2.append(tempD2)
#     # pre-emptively just set t to be what it should, calc mean and var of roughness
#     TimeList = np.mean(TimeList, 0)
#     roughMean = np.mean(RoughnessList, 0)
#     roughVar = np.var(RoughnessList, 0)
#     # calculate roughness
#     return np.array(PerimeterList), np.array(AreaList), roughMean, roughVar, dist, dist2, np.array(TimeList)

