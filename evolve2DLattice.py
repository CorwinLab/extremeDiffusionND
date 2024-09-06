import numpy as np
import os

from scipy.ndimage import morphology as m
import csv
import npquad
import pandas as pd
import sys
import glob

def changeArraySize(array, size, fillval):
	"""
	Takes an existing numpy array and expands it to the size you want to change it to with a fill value
	:param array: the array you want to change the size of
	:param size: the new size (as in LxL); to double array, put 2*array.shape[0]
	:param fillval: what you want to fill the new entries with
	:return: newArray: the old array but expanded to the specified size, with the new entried filled
	"""
	length = (array.shape[0]) // 2
	newsize = size // 2
	if length < size:  # make sure you're always up-sizing
		newArray = np.full((2 * newsize + 1, 2 * newsize + 1), fillval, dtype=array.dtype)
		newArray[newsize - length:newsize + length + 1, newsize - length:newsize + length + 1] = array
	else:  # if new size is smaller than old size, do nothing
		newArray = array
	return newArray

def getRandVals(distribution, rng, shape, params):
	"""
	Get random values across the lattice with specified shape according to the specified distribution
	:param distribution: specify 'uniform', 'SSRW', 'dirichlet' to pull biases for moving each direction
	:param rng: numpy random number generator (should be np.random.default_rng() passed in)
	:param shape: how many values you want to pull and in what format
	:param params: specifically for 'dirichlet' specify the parameters (i.e alpha), otherwise should be None
	:return biases: the array of random values from specified distribution, to be used as probability in each dir.
	"""
	if distribution == 'uniform':  # alpha = 1, hence no parameters
		biases = rng.dirichlet([1] * 4, shape)
	elif distribution == 'SSRW':
		biases = np.full([shape, 4], 1 / 4)
	elif distribution == 'dirichlet':
		# params should be a number >0 (it's alpha from dirichlet dist.)
		params = float(params)
		biases = rng.dirichlet([params] * 4, shape)
	else:  # catchall in case inputted distribution isn't one of these.
		# maybe take out the print statement once i move to talapas?
		print(f"Specified distribution: {distribution} is not an option. 'uniform', 'SSRW', or 'dirichlet' allowed."
			  "\n Default to SSRW Distribution")
		biases = np.full([shape, 4], 1/4)
	return biases

def executeMoves(occupancy, i, j, rng, distribution, isPDF, distributionParams=None):
	"""
	Evolves 2Dlattice according to a distribution, with the option to evolve agents or the PDF
	:param occupancy: the array you are working in or the initial occupancy
	:param i, j = sites with agents in them/sites that are occupied
	:param rng: numpy random number generator (should be np.random.default_rng() passed in)
	:param distribution: string specifying distribution you are using to generate biases
	:param isPDF: boolean; if true then multiplies biases; if false then draws multinomial\
	:param distributionParams: the parameters of the distribution
	:return occupancy: the new occupancy after moving everything
	"""
	# Generate biases for each site
	biases = getRandVals(distribution, rng, i.shape[0], distributionParams)
	# On newer numpy we can vectorize to compute the moves
	if isPDF:  # if doing PDF then multiply by biases as npquads
		moves = occupancy[i, j].reshape(-1, 1) * biases.astype(np.quad)
	else:
		if i.shape[0] == 1:
			moves = rng.multinomial(occupancy[i,j].astype(int),biases.squeeze())
		else:
			moves = rng.multinomial(occupancy[i, j].astype(int), biases)
	# Note that we can use the same array because we're doing checkerboard moves
	# If we want to use a more general jump kernel we need to use a new (empty) copy of the space
	occupancy[i, j - 1] += moves[:, 0]  # left
	occupancy[i + 1, j] += moves[:, 1]  # down
	occupancy[i, j + 1] += moves[:, 2]  # right
	occupancy[i - 1, j] += moves[:, 3]  # up
	occupancy[i, j] = 0  # Remove everything from the original site, as it's moved to new sites
	return occupancy

def evolve2DLattice(occupancy, maxT, distribution, params, isPDF, startT=1,
					rng=np.random.default_rng(), boundary=None):
	"""
	generator; evolves agents or PDF in a 2D lattice out to some time maxT with dynamic scaling.
	:param occupancy: either number (NParticles) or occupancy array
	:param maxT: timestep you want to go out to
	:param distribution: string, specify the distribution of biases; see getRandVals for options
	:param params: parameters of distribution
	:param isPDF: boolean; if true then multiplies biases; if false then evolves agents
	:param startT: optional; time you want to start at; default 1
	:param rng: the numpy random number generator obj(default np.random.default_rng() )
	:param boundary: (numpy array of bools) boundary conditions should be same size as occupancy.
		Default set to None, which means the boundary is OFF
	:return t: yield time
	:return occupancy: yield occupancy array
	"""
	for t in range(startT, maxT):
		# since boundary is None is an option, find occupied sites appropriately
		if boundary is None:  # no boundary
			i, j = np.where(occupancy != 0)
		else:  # there is a boundary
			i, j = np.where((occupancy != 0) & boundary)
		# Only consider doubling if we're not using a boundary array
		if not (isinstance(boundary, np.ndarray)):
			# If the occupied sites are at the limits (i.e if min(i,j) = 0 or max(i,j) = size)
			# then we need to enlarge occupancy and create a new array.
			if ((np.min([i, j]) <= 0)
					or (np.max([i, j]) >= np.min(occupancy.shape) - 1)):
				occupancy = changeArraySize(occupancy, 2*occupancy.shape[0], fillval=0)
				# These next two lines are a waste; we could just do index translation
				i, j = np.where(occupancy != 0)  # inside if that requires no boundary, so don't need to check it
		occupancy = executeMoves(occupancy, i, j, rng, distribution, isPDF, params)
		yield t, occupancy

def prepareBoundary(maxT, absorbingRadius):
	"""
	subfunction to generate the absorbing boundary mask/array for evovleAgents or evolvePDF
	:param maxT: inherit the maximum time from evolveAgents/PDF
	:param absorbingRadius: inherit from evolveAgents/PDF; should be None or an integer
	:return distToOrigin <=absorbingRadius: the mask that is boundary
	"""
	# get the distance to Origin of lattice
	# temp = np.arange(-maxT, maxT+1)
	# x, y = np.meshgrid(temp, temp)
	# distToOrigin = np.sqrt(x**2 + y**2)

	# geneerate distToOrigin using maxT=L
	x, y, distToOrigin = getDistFromCenter(maxT)

	# make sure to check that the absorbingRadius is at largest, the lattice size
	if absorbingRadius >= (maxT - 1):
		absorbingRadius = maxT - 1  # if it's larger than lattice, set it to be what it should be
	return distToOrigin <= absorbingRadius  # 12 July 2024: changed to not return new radius

def organizePDFStats(occ, integratedPDF, listOfNs, firstPDFStats, firstIntegratedPDFStats):
	"""
	reshape and combine all PDF & integratedPDF stats into arrays as we evovle in time
	:param t, occ: the t, occ in the evolve2DLattice generator
	:param integratedPDF: pass in the integratedPDF which is +=occ at every timestep
	:param listOfNs: array; values of N at which you want to calc. roughness stats
	:param firstPDFStats: the initial stats that start the picket fence (?) needs
	:param firstIntegratedPDFStats: the initial stats that start the picket fence (?) needs
	:return: PDFstats, integratedPDFStats: two arrays;shape is (time, N's, stats); statsN0 = stats[:,0,:]
		stats are: perimeter, area, roughness, radius moment 1,radius moment 2, N
	"""
	# update the integratedPDF stats
	tempIntegratedPDFStats = np.expand_dims(np.array([getContourRoughness(integratedPDF, N) for N in listOfNs]), axis=0)
	integratedPDFStats = np.vstack((firstIntegratedPDFStats, tempIntegratedPDFStats))
	# update the PDF stats
	rolledPDF = occ + np.roll(occ, 1)
	tempPDFstats = np.expand_dims(np.array([getContourRoughness(rolledPDF, N) for N in listOfNs]), axis=0)
	PDFstats = np.vstack((firstPDFStats, tempPDFstats))  # shape is like (time, N's, stats); statsN0 = stats[:,0,:]
	return PDFstats, integratedPDFStats

def organizeTArrivalStats(t, tArrival, firstStats):
	"""
	reshape and combine the tArrival roughness stats into one array as lattice is evolved
	:param t: the t in the evolve2D lattice generator
	:param tArrival: array of tArrivals from evolveAgents
	:param firstStats: the initial stats that start the picket fence (?)
	:return: stats: an array of shape (time, stat) and the stats are: perimeter, area, roughness,
		radius moment1, randius moment 2, tau (time)
	"""
	tempstats = np.array(getTArrivalRoughness(tArrival, t))
	stats = np.vstack((firstStats, tempstats))
	return stats

def getListOfTimes(maxT, startT=1, num=500):
	"""
	Generate a list of times, with approx. 10 times per decade (via np.geomspace), out to time maxT
	:param maxT: the maximum time to which lattice is being evolved
	:param startT: initial time at which lattice evolution is started
	:param num: number of times you want
	:return: the list of times
	"""
	return np.unique(np.geomspace(startT, maxT, num=num).astype(int))

def getListOfNs():
	"""
	return a list of particle numbers between 100 and 1e301
	:return: list of particle numbers, about going every 2 from 1e2 to 1e10, and then every 50 from 1e50 to 1e301
	"""
	# to get n = 1e2 - 1e10, then in increments of 50
	return 10. ** np.hstack([np.arange(2, 11, 2), np.arange(50, 301, 50)])

def evolvePDF(maxT, distribution, params, startT=1, absorbingRadius=None, boundaryScale=5, listOfTimes=None, listOfNs=None):
	"""
	Evolves a RWRE PDF and integrated PDF on a 2DLattice with dynamic scaling.
	Does not return tArrivals b/c theyre deterministic
	:param maxT: timestep you want to go out to (integer)
	:param distribution: string; specify distribution from which biases pulled
	:param params: the parameters of the specified distribution
	:param startT: optional arg, starts at 1
	:param absorbingRadius: optional parameter to set location of absorbing boundary condition
		if absorbingRadius <0 >then turns it off; default=None which sets the radius to scale with
		boundaryScale*np.sqrt(maxT)
	:param boundaryScale: the coeffence before sqrt(maxT) that sets the default boundary scaling
	:param listOfTimes: array, list of times at which statistics calculated
	:param listOfNs: array, list of occupancies/particle numbers for which statistics calculated
	:return occ: the final evolved occupancy array
	:return integratedPDF: the final cumulative probabiliity
	:return integratedPDFStats: the roughness stats of the CDF for list of times and list of Ns
	:return listofTimes: the times generated/used at which to roughness stats calculated
	"""
	occupancy = np.zeros((2*maxT+1, 2*maxT+1), dtype=np.quad)  # initialize occupancy
	occupancy[maxT, maxT] = 1
	# No boundary mask
	if absorbingRadius < 0:
		absorbingBoundary = None  # set absorbingBoundary to be none
	# Boundary mask
	else:
		if absorbingRadius is None:  # if no radius specified, make it this default:
			absorbingRadius = boundaryScale * np.sqrt(maxT)
		# get boundary
		absorbingBoundary = prepareBoundary(maxT, absorbingRadius)
	# Initialize needed variables
	integratedPDF = np.copy(occupancy) # should inehrit occupancy's dtype
	if listOfTimes is None:
		listOfTimes = getListOfTimes(maxT, startT=startT,num=round(10 * np.log10(maxT))).astype(int)
	if listOfNs is None:
		listOfNs = getListOfNs()
	# Run the data and stats generation loop
	for t, occ in evolve2DLattice(occupancy, maxT, distribution, params, True, boundary=absorbingBoundary):
		integratedPDF += occ
		if t == startT+1:  # initialize the stats
			integratedPDFStats = np.expand_dims(np.array([getContourRoughness(integratedPDF, N) for N in listOfNs]), axis=0)
			rolledPDF = occ + np.roll(occ,1)
			PDFstats = np.expand_dims(np.array([getContourRoughness(rolledPDF,N) for N in listOfNs]), axis=0)
		elif t in listOfTimes:  # so we don't double count t == 2
		# shape is (time, N's, stats); statsN0 = stats[:,0,:]
			PDFstats, integratedPDFStats = organizePDFStats(occ, integratedPDF, listOfNs, PDFstats, integratedPDFStats)
	return occ, integratedPDF, np.array(PDFstats), np.array(integratedPDFStats), listOfTimes, absorbingBoundary

def evolveAgents(occupancy, maxT, distribution, params, startT=1, absorbingRadius=False, listOfTimes=None):
	"""
	Evolves a 2DLattice with dynamic scaling.
	:param occupancy: initial occupancy, can be a number (NParticles) or an existing array
	:param maxT: timestep you want to go out to (integer)
	:param distribution: string; specify distribution from which biases pulled
	:param params: the parameters of the specified distribution
	:param startT: optional arg, starts at 1
	:param absorbingRadius: optional parameter to set location of absorbing boundary condition
	:param listOfTimes: array, list of times at which statistics calculated
	:return occ: the final evolved occupancy array
	:return tArrival: the array with the time of first arrival for every site in the occupancy array
	"""
	notYetArrived = np.nan
	if np.isscalar(occupancy):  # if given a scalar (ie NParticles or something), initializes array
		NParticles = occupancy
		occupancy = np.array([[NParticles]], dtype=float)  # NO QUADS IN AGENTS
	# no boundary
	if absorbingRadius < 0:
		absorbingBoundary = None
	# there is a boundary
	else:  # if there is a boundary, initialize occupancy as fixed lattice size
		occupancy = np.zeros((2*maxT+1,2*maxT+1), dtype=float)
		occupancy[maxT, maxT] = NParticles
		if absorbingRadius == False:
			absorbingRadius = 2 * np.sqrt(maxT * np.log(NParticles))  # calculate absorbingRadius
		absorbingBoundary = prepareBoundary(maxT, absorbingRadius)  # get mask for the boundary
	# initialize necessary arrays and variables
	tArrival = np.copy(occupancy)  # inherits floats
	tArrival[:] = notYetArrived
	tArrival[occupancy > 0] = 0
	if listOfTimes is None:
		listOfTimes = getListOfTimes(maxT, startT=startT,num=round(10 * np.log10(maxT))).astype(int)
	# data and stats generation
	for t, occ in evolve2DLattice(occupancy, maxT, distribution, params, False, boundary=absorbingBoundary):
		if tArrival.shape[0] != occ.shape[0]:
			# Note: this is fragile, we assume that doubling tArrival will always work
			tArrival = changeArraySize(tArrival, 2*tArrival.shape[0], fillval=notYetArrived)
		tArrival[(occ > 0) & (np.isnan(tArrival))] = t  # update tArrival array
		if t == startT+1:  # initialize stats
			stats = np.array(getTArrivalRoughness(tArrival, t))  # p, a, r, d, d2, t
		elif t in listOfTimes:
			stats = organizeTArrivalStats(t, tArrival, stats)
	return tArrival, occ, np.array(stats), absorbingBoundary

def run2DAgent(*args, **kwargs):
	"""
	copy of run2DAgent but using *args and **kwargs instead
	:param args:
	:param kwargs:
	:return:
	"""
	for t, occ in evolve2DLattice(*args,**kwargs):
		pass
	return t, occ

def getTArrivalMeanAndVar(path):
	"""
	Takes a directory filled with tArrival arrays and finds the mean and variance of tArrivals
	Calculates progressively, since loading in every array will use too much memory
	:param path: the path of the directory
	:return: finalMom1: the first moment (mean) of tArrivals
	:return finalMom2 - finalMom1**2: the variance of the tArrivals
	:return goodData: the mask, so you only look at the stuff where every agent has gotten to
	"""
	filelist = sorted(os.listdir(path))
	notYetArrived = np.nan
	# initialize the moments & mask
	tArrMom1, tArrMom2, goodData = None, None, None  # moment 1 is just t, moment 2 is t**2
	# go through each file and pull out the tArrival array
	for file in filelist:
		tArrival = np.load(f'{path}/{file}')['tArrival']
		if tArrMom1 is None:  # if you are on the first file, make the moment arrays using the first file
			tArrMom1 = tArrival
			tArrMom2 = tArrival ** 2
			# we want the non-nans in this array to be True since we're masking out the places where
			# only one or two walkers have been
			goodData = np.invert(np.isnan(tArrival))
		else:  # check that array sizes agree; make them the same if they don't
			if tArrMom1.shape[0] < tArrival.shape[0]:
				tArrMom1 = changeArraySize(tArrMom1, tArrival.shape[0], fillval=notYetArrived)
				tArrMom2 = changeArraySize(tArrMom2, tArrival.shape[0], fillval=notYetArrived)
				goodData = changeArraySize(goodData, tArrival.shape[0], fillval=False)
			if tArrMom1.shape[0] > tArrival.shape[0]:
				tArrival = changeArraySize(tArrival, tArrMom1.shape[0], notYetArrived)
			# now cumulatively add the moments
			goodData *= np.invert(np.isnan(tArrival))
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

def getStateRoughness(binaryImage):
	"""
	function that takes in a binary image (bools) & calcs perimeter area roughness etc
	:param binaryImage: 2d binary image (tArrival  or occ or something)
	:return: perimeter, area, roughness, radius moment1, radius moment2: instantaneous measurements
		of the inputted state (binary image)
	"""
	edge = (binaryImage ^ m.binary_erosion(binaryImage))  # find edge of image
	# get radius of each point from origin
	i, j = np.where(edge)
	L = binaryImage.shape[0] // 2
	# subtract L from i and j here because the binary image that was input should be like O to 2L in dimension
	# but when we calculate polar coordinatese we want the center of the array to be at the origin
	i, j = i - L, j - L
	radius, theta = cartToPolar(i, j)
	# get moments of radius
	radiusMoment1 = np.mean(radius)
	radiusMoment2 = np.mean(np.square(radius))
	# calculate the roughness of the state
	perimeter = np.sum(edge)
	area = np.sum(binaryImage)
	try:  # if we are looking at contours where max(PDF val) < 1/N then this should be 0/0
		roughness = perimeter/np.sqrt(area)
	except:  # so fill roughness with a nan instead
		roughness = np.nan
	# return perimeter, area, roughness, etc of the state
	return perimeter, area, roughness, radiusMoment1, radiusMoment2


def getTArrivalRoughness(tArrival, tau):
	""" Calculate the surface roughness of tArrivals as a function of time by (#edge pixels)/(total pixels reached)^1/2
	Takes one tArrival array and roughness stats. as function of time tau
	:param tArrival: the tArrival array, going out to its max time
	:param tau: time for which roughness being calculated
	:return instantaneousStats: np array with (perimeter, area, roughness, radius moment 1, radius moment2, tau
	"""
	# initialize the instantaneousStats w/ the stats at tau=1
	mask = (tArrival <= tau) & (np.invert(np.isnan(tArrival)))
	# use getStateRoughness to then calc stats and return them plus time=tau
	return getStateRoughness(mask)+(tau,)  # p, a, r, d, d2, tau

def getContourRoughness(Array, N):
	"""
    Calculate roughness of an evolved CDF as a function of 1/NParticles by looking at contour of
    probability >= 1/N. implicitly also a function of time goes inside the evolve2Dlattice generator loop
    :param Array: 2D array
    :param N: the number of particles for which you want probability to be >= 1/N
    :return: perimeter, area, roughness, radius moment 1, adius moment 2, N
    """
	mask = (Array >= np.quad(1/N))  # binary image
	roughnessStats = getStateRoughness(mask)  # get roughness vals for that specific 1/N
	return roughnessStats + (N,)  # p, a, r, d, d2, N

def getRoughnessMeanVar(path):
	"""
	Calculate the mean and variance of the roughness statistics (tArrStats, pdfStats, and
	integratedPDFStats). since these are small arrays we can just load them in all at once
	and use np.mean and np.var
	:param path: directory of stored data
	:return: meanStats, varStats: mean and variance of tArrival roughness stats (perimeter, area,
		roughness, radius moment1, radius moment 2) as from evolveAgents
	:return: meanPDFStats, varPDFStats: mean and var of PDF contour roughness stats as from evolvePDF
	:return: meanIntegratedPFStats, varIntegratedPDFStats: mean and var of integrated PDF Contour roughness stats
		which are also still perimeter, area, roughness, radius moment1, radius moment2, N; from evolvePDF
	"""
	filelist = sorted(os.listdir(path))
	temp = np.load(f'{path}/{filelist[0]}')  # load in 1 so we can check keywords of the npz files
	if 'tArrStats' or 'tArrivalStats' in temp.files:  # if loading tArrival Stats
		# throw all stats into one array, so we can take mean and var
		listOfStats = np.array([np.load(f'{path}/{filelist[i]}')['tArrStats'] for i in range(len(filelist))])
		meanStats = np.mean(listOfStats, 0)  # along axis 0 so we avg. over the files
		varStats = np.var(listOfStats, 0)
		return meanStats, varStats  # return appropriate mean & var
	else:  # if loading PDF & integratedPDF
		# throw all stats into one array to take mean and var
		listOfPDFStats = np.array([np.load(f'{path}/{filelist[i]}')['pdfStats'] for i in range(len(filelist))])
		listOfIntegratedPDFStats = np.array([np.load(f'{path}/{filelist[i]}')['integratedPDFStats'] for i in range(len(filelist))])
		meanPDFStats = np.mean(listOfPDFStats, 0)  # PDF; along axis 0 to avg. over files
		varPDFStats = np.var(listOfPDFStats, 0)
		meanIntegratedPDFStats = np.mean(listOfIntegratedPDFStats)  # integratedPDF
		varIntegratedPDFStats = np.var(listOfIntegratedPDFStats)
		return meanPDFStats, varPDFStats, meanIntegratedPDFStats, varIntegratedPDFStats  # return appropriate mean & var

def getDistFromCenter(L):
	""" Take an occupancy array (or a numpy array) and create a meshgrid to get
	the distance to center of every lattice point
	:param occ: np array
	:return x, y: the meshgrid
	:return dist_from_center: np array with the distance from the center of each site as the val.
	"""
	temp = np.arange(-L, L+1)
	x, y = np.meshgrid(temp, temp)
	return x, y, np.sqrt(x**2+y**2)

#TODO: remove dependence on getDistFromCenter here because I want to do it before
# i call all of these get()Mask functions
# so getDistFromCenter returns x, y, and distFromCenter....
# and I need get()Mask to take it distFromCenter
def getInsideSphereMask(dist, r):
	mask = (dist < r)
	return mask


def getLineMask(x, y, r, axis=0):
	'''something here
	:param occ: pass in occupancy array, should be np array
	:param r: value at which the lines are
	:param axis: if 0 (default), line is vertical, get everything to the right of it;
		if 1, line horizontal and get everything past it
	:return indices: indices of occ which fulfill x or y >= r
	'''
	if axis == 0:  # if asking for vertical line
		mask = (x >= r)
	else:  # if asking for horizontal line
		mask = (y >= r)
	return mask

def getBoxMask(x, y, line):
	'''
	use np.where in the x direction and y direction to get a box?
	'''
	mask = ((x>=line ) & (y >= line))
	return mask

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

	x, y, dist_to_center = getDistFromCenter(L)
	boundary = (dist_to_center <= R)

	# pass in the x, y, and sqrt(x^2+y^2) from the getDistFromCenter meshgrid
	masks = [getInsideSphereMask(dist_to_center, r) for r in Rs]
	line_masks = [getLineMask(x, y, r) for r in Rs]
	# ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
	ts = getListOfTimes(1,tMax)  # default num is 500
	# Need to make sure occ doesn't change size
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, True, boundary=boundary):
		# Get probabilities inside sphere
		if t in ts:
			probs = [1 - np.sum(occ[mask]) for mask in masks]
			writer.writerow([t, *probs])
			f.flush()

			# Get probabilities outside line
			probs = [np.sum(occ[mask]) for mask in line_masks]
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

	vs : np array
		list of velocities at which boundary moves?
	
	Example
	-------
	tMax = 100
	L = 250
	R = L-1
	vs = np.array([5, 10])  # should be about a decade or two of difference
	savefile = 'PDF.txt'
	linefile = 'Line.txt'
	distribution = 'dirichlet'
	params = 1/10
	measureAtVsOnSphere(tMax, L, R, vs, distribution, params, savefile, linefile)
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

	x, y, dist_to_center = getDistFromCenter(L)
	boundary = (dist_to_center <= R)

	# ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
	ts = getListOfTimes(1,tMax)  #default is 500
	# Need to make sure occ doesn't change size
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, True, boundary=boundary):
		# Get probabilities inside sphere
		if t in ts: 
			Rs = list(np.array(vs * t).astype(int))

			# pass in the x, y, and sqrt(x^2+y^2) from the getDistFromCenter meshgrid
			masks = [getInsideSphereMask(dist_to_center, r) for r in Rs]
			line_masks = [getLineMask(x, y, r) for r in Rs]

			probs = [1-np.sum(occ[mask]) for mask in masks]
			writer.writerow([t, *probs])
			f.flush()

			# Get probabilities outside line
			probs = [np.sum(occ[mask]) for mask in line_masks]
			writer_line.writerow([t, *probs])
			f_line.flush()

	f_line.close()
	f.close()


def measureLineProb(tMax, L, R, vs, distribution, params, saveFile):
	'''
	Parameters
	----------
	L : int 
		Radius of size of box

	tMax : int 
		Maximum time to iterate to

	R : float
		Radius of circle for circular boundary conditions

	vs : nummpy array
			something here
	Example
	-------
	tMax = 100
	L = 250
	R = L-1
	vs = np.array([0.01,0.1])
	linefile = 'Line.txt'
	distribution = 'dirichlet'
	params = 1/10
	measureOnSphere(tMax, L, R, Rs, distribution, params, linefile)
	'''
	# ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
	ts = getListOfTimes(1,tMax)  # num default is 500
	vs = np.array(vs)

	write_header = True
	if os.path.exists(saveFile):
		data = pd.read_csv(saveFile)
		max_time = max(data['Time'].values)
		if max_time == ts[-2]:
			print(f"File Finished{f}", flush=True)
			sys.exit()
		ts = ts[ts > max_time]
		print(f"Starting at: {ts[0]}", flush=True)
		write_header = False

	# Set up writer and write header if save file doesn't exist
	f = open(saveFile, 'a')
	writer = csv.writer(f)
	if write_header:
		# This was previously misnamed Mean(Sam) and Var(Sam)
		writer.writerow(["Time", *vs])

	# Create occupancy array, goes from 0 to 2L+1 in each dir.
	occ = np.zeros((2 * L + 1, 2 * L + 1))
	occ[L, L] = 1
	
	# x = np.arange(-L, L+1)
	# xx, yy = np.meshgrid(x, x)
	# dist_to_center = np.sqrt(xx ** 2 + yy ** 2)
	x, y, dist_to_center = getDistFromCenter(L)
	boundary = (dist_to_center <= R)
	
	# Need to make sure occ doesn't change size
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, 
								  True, boundary=boundary):
		# Get probabilities inside sphere
		if t in ts:
			Rs = np.array(vs * t).astype(int)

			# np.sum (axis=1) will sum each column, so you'll get 1d array [sum column 1, sum column 2,...]
			# np.cumsum will give [sum c1, sumc1+sumc2, sumc1+sumc2+sumc3, ...] (going from ~0 to 1)
			# np.flip turns it into [1, ...., 0], which we want for precision reasons.
			# this means we will be measuring to the left of our lines
			xCDF = np.flip(np.cumsum(np.sum(occ, axis=1)))

			# next we get the indices to the left of the line at R?
			# You add L to the Rs to shift it to the center of the array
			indices = Rs + L  # this is faster than np.where which is what getLineIndices uses

			# eventaully Rs+L > 2L+1
			indices_outside_array = indices >= len(xCDF)
			indices[indices_outside_array] = 0

			probs = xCDF[indices]
			probs[indices_outside_array] = 0
			
			# Don't do this line
			# probs = [np.sum(occ[xx >= r]) for r in Rs]
			writer.writerow([t, *probs])
			f.flush()

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
	
	# x = np.arange(-L, L+1)
	# xx, yy = np.meshgrid(x, x)
	# dist_to_center = np.sqrt(xx ** 2 + yy ** 2)
	x, y, dist_to_center = getDistFromCenter(L)
	boundary = dist_to_center <= R

	# ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
	ts = getListOfTimes(1,tMax)  # default num=500
	# Need to make sure occ doesn't change size
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, True, boundary=boundary):
		# Get probabilities inside sphere
		if t in ts: 
			Rs = list(np.array(1/2 * t**(np.array(alpha))).astype(int))

			# pass in the x, y, and sqrt(x^2+y^2) from the getDistFromCenter meshgrid
			masks = [getInsideSphereMask(dist_to_center, r) for r in Rs]
			line_masks = [getLineMask(x, y, r) for r in Rs]

			probs = [1-np.sum(occ[mask]) for mask in masks]
			writer.writerow([t, *probs])
			f.flush()

			# Get probabilities outside line
			probs = [np.sum(occ[mask]) for mask in line_masks]
			writer_line.writerow([t, *probs])
			f_line.flush()

	f_line.close()
	f.close()

# guess i'm writing code like jacob now
def measureAtVsBox(tMax, L, R, vs, distribution, params, barrierScale,
				   boxSaveFile, vLineSaveFile, sphereSaveFile):
	'''
	string here
	'''
	# initialize occ with absorbing boundary,
	occ = np.zeros((2 * L + 1, 2 * L + 1),dtype=np.quad)
	occ[L, L] = 1
	#ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))  # generate times
	ts = getListOfTimes(1,tMax)  # default num=500

	# get meshgrid stuff since the size of occ never changes
	x, y, dist_to_center = getDistFromCenter(L)
	absorbingBoundary = (dist_to_center <= R)

	# check if savefiles exist first
	if os.path.exists(boxSaveFile):
		data = pd.read_csv(boxSaveFile)
		max_time = max(data['Time'].values)
		# if file exists and is finished, exit
		if max_time == ts[-2]:
			print(f"File Finished", flush=True)
			sys.exit()

	# Set up writer and write header if save file doesn't exist,
	with open(boxSaveFile, 'w') as f, open(vLineSaveFile, 'w') as f_vline, open(sphereSaveFile, 'w') as f_sphere:
		# create writers
		writer = csv.writer(f)  # prob. in quadrant
		writer_vline = csv.writer(f_vline)  # prob. past vertical line
		writer_sphere = csv.writer(f_sphere)  # prob outside sphere
		# write data (since we switched from 'a' to 'w' this is fine)
		writer.writerow(["Time", *vs])
		writer_vline.writerow(['Time', *vs])
		writer_sphere.writerow(["Time", *vs])

		# generate data
		for t, occ in evolve2DLattice(occ, tMax, distribution, params, True, boundary=absorbingBoundary):
			# Get probabilities inside sphere
			if t in ts:
				RsScale = eval(barrierScale)

				# remove the astype(int) because it's causing the data to "turn on"
				# at weird spots, and is unnecessary because get(insertshape)Indices already
				# implicitly takes into account the lattice spacing.
				Rs = list(np.array(vs * RsScale))

				# grab indices for box, past lines, and outside sphere
				# pass in the x, y, and sqrt(x^2+y^2) from the getDistFromCenter meshgrid
				box_masks = [getBoxMask(x, y, r) for r in Rs]  # should be a list of masks
				vline_masks = [getLineMask(x, y, r, axis=0) for r in Rs]  # past vertical line
				sphere_masks = [getInsideSphereMask(dist_to_center, r) for r in Rs]  # outside sphere

				# get probabilities in quadrant
				probs = [np.sum(occ[mask]) for mask in box_masks]
				writer.writerow([t, *probs])
				f.flush()

				# get probabilities past veritcal line:
				probs = [np.sum(occ[mask]) for mask in vline_masks]
				writer_vline.writerow([t,*probs])
				f_vline.flush()

				#get probabilities outside sphere
				probs = [1-np.sum(occ[mask]) for mask in sphere_masks]
				writer_sphere.writerow([t,*probs])
				f_sphere.flush()

		f_sphere.close()
		f_vline.close()

def getQuadrantMeanVar(path, filetype, tCutOff=None,takeLog=True):
	"""
	Takes a directory filled  arrays and finds the mean and variance of cumulative probs. past various geometries
	Calculates progressively, since loading in every array will use too much memory
	:param path: the path of the directory, /projects/jamming/fransces/data/quadrant/distribution/tMax
	:param filetype: string, 'box', 'hline', 'vline, 'sphere'
	:param tCutOff: default mmax val of time; otherwise the time at which you want to cut off the data to look at
	:return: finalMom1: the first moment (mean) of log(probabilities)
	:return finalMom2 - finalMom1**2: the variance of log(probabilities)
	"""
	# grab the files in the data directory that are the Box data
	if filetype == 'box':
		files = glob.glob("Box*",root_dir=path)
	elif filetype == 'vline':
		files = glob.glob("vLine*", root_dir=path)
	elif filetype == 'hline':
		files = glob.glob("hLine*",root_dir=path)
	elif filetype == 'sphere':
		files = glob.glob("sphere*",root_dir=path)
	# initialize the moments & mask, fence problem
	# moment1, moment2 = None, None
	firstData = pd.read_csv(f"{path}/{files[0]}")
	if tCutOff is None:
		tCutOff = np.max(firstData['Time'])
	firstData = firstData[firstData.Time <= tCutOff]
	if takeLog:
		firstData = np.log(firstData.values)
	else:
		firstData = firstData.values
	moment1, moment2 = firstData, np.square(firstData)
	# load in rest of files to do mean var calc, excluding the 0th file
	for file in files[1:]:
		data = pd.read_csv(f"{path}/{file}")
		data = data[data.Time <= tCutOff]
		if takeLog:
			data = np.log(data.values)
		else:
			data = data.values
		moment1 += data
		moment2 += np.square(data)
	moment1 = moment1 / len(files)
	moment2 = moment2 / len(files)
	# Return the mean and the variance
	# note this also will take the mean and var of time. what you want is
	# meanBox[:,1:] to get just the probs.
	return moment1, moment2 - np.square(moment1)
