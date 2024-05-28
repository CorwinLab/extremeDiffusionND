import numpy as np
import os
from scipy.ndimage import morphology as m
import csv
import npquad
import pandas as pd
import sys

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
	if distribution == 'uniform':  # alpha = 1, hence no parameters
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
	biases = getRandVals(distribution, rng, i.shape[0], params).astype(np.quad)
	# print('Generated biases', occupancy.dtype, biases.dtype)
	# On newer numpy we can vectorize to compute the moves
	if PDF:  # if doing PDF then multiply by biases
		moves = occupancy[i, j].reshape(-1, 1) * biases
		#print(moves.dtype)
		# occupancy = occupancy.astype(np.float64)  # since the PDF requires floats but the agents require ints, cast correctly
		# moves = occupancy[i, j].reshape(-1, 1) * biases  # reshape -1 takes the shape of occupancy
	else:
		occupancy = occupancy.astype(int) # since the PDF requires floats but the agents require ints, cast correctly
		moves = rng.multinomial(occupancy[i, j], biases)
	# Note that we can use the same array because we're doing checkerboard moves
	# If we want to use a more general jump kernel we need to use a new (empty) copy of the space
	occupancy[i, j - 1] += moves[:, 0]  # left
	occupancy[i + 1, j] += moves[:, 1]  # down
	occupancy[i, j + 1] += moves[:, 2]  # right
	occupancy[i - 1, j] += moves[:, 3]  # up
	occupancy[i, j] = 0  # Remove everything from the original site, as it's moved to new sites
	# print('Finished w/ occupancy')
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
		newArray = np.full((2 * newsize + 1, 2 * newsize + 1), fillval, dtype=array.dtype)
		newArray[newsize - length:newsize + length + 1, newsize - length:newsize + length + 1] = array
	else:
		newArray = array
	return newArray


# main functions & generators + wrappers
def evolve2DLattice(occupancy, maxT, distribution, params, PDF, startT=1,
					rng=np.random.default_rng(), boundary=True):
	"""
	generator; evolves agents or PDF in a 2D lattice out to some time maxT with dynamic scaling.
	:param occupancy: either number (NParticles) or occupancy array
	:param maxT: timestep you want to go out to
	:param distribution: string, specify the distribution of biases
	:param params: parameters of distribution
	:param PDF: boolean; if true then multiplies biases; if false then evolves agents
	:param startT: optional; time you want to start at; default 1
	:param rng: the numpy random number generator obj(default np.random.default_rng() )
	:param boundary: (numpy array of bools) boundary conditions should be same size as occupancy
	:return t: yield tArrival array
	:return occupancy: yield occupancy array
	"""
	for t in range(startT, maxT):
		# Find the occupied sites
		i, j = np.where((occupancy != 0) & boundary)
		if np.max(i) == occupancy.shape[0]-1:
			print("You hit the edge of the occupancy array")
		# Only consider doubling if we're not using a boundary array
		if not (isinstance(boundary, np.ndarray)):
			# If the occupied sites are at the limits (i.e if min(i,j) = 0 or max(i,j) = size)
			# then we need to enlarge occupancy and create a new array.
			if ((np.min([i, j]) <= 0)
				or (np.max([i, j]) >= np.min(occupancy.shape) - 1)):
				occupancy = doubleArray(occupancy, occupancy.dtype)
				# These next two lines are a waste; we could just do index translation
				sites = (occupancy != 0)
				i, j = np.where(sites & boundary)
		occupancy = executeMoves(occupancy, i, j, rng, distribution, PDF, params)
		# executeMoves(occupancy, i, j, rng, distribution, PDF, params)
		# print('getting ready to yield')
		yield t, occupancy


# 22 May 2024: generateFirstArrival --> evolvePDF and evolveAgents
#TODO: either move occupancy as an optional parameter, or nix it entirely
def evolvePDF(maxT, distribution, params, startT=1, absorbingRadius = None):
	"""
	Evolves a RWRE PDF on a 2DLattice with dynamic scaling.
	Does not return tArrivals b/c theyre deterministic
	:param maxT: timestep you want to go out to (integer)
	:param distribution: string; specify distribution from which biases pulled
	:param params: the parameters of the specified distribution
	:param startT: optional arg, starts at 1
	:param absorbingRadius: optional parameter to set location of absorbing boundary condition
	:return occ: the final evolved occupancy array
	"""
	notYetArrived = np.nan #-1
	if distribution == 'dirichlet':
		params = float(params)
	else:
		params = None
	#TODO: Eventually put this lil bit into a function
	L = (maxT-1)//2  # if maxT = 10000, then this should be 4999
	temp = np.arange(-L, L + 1)
	x, y = np.meshgrid(temp, temp)
	distToOrigin = np.sqrt(x**2 + y**2)
	if absorbingRadius is None:  # establish absorbing boundary
		absorbingRadius = L - 1
		absorbingBoundary = (distToOrigin <= absorbingRadius)
	occupancy = np.zeros((2 * L + 1, 2 * L + 1),dtype=np.quad)  # initialize occupancy
	occupancy[L, L] = 1
	if absorbingRadius == 'off':  # this is a stupid way to get the goddamn diamond shapes
		occupancy = np.zeros((2*maxT+1,2*maxT+1), dtype=np.quad)  # basically make the occupancy big enough to see the shape
		occupancy[maxT,maxT] = 1
		absorbingBoundary = True  # and turn off the boundary otherwise ):
	else:
		absorbingBoundary = (distToOrigin <= absorbingRadius)
	CDF = np.copy(occupancy)# .astype(np.quad)  # iniitalize CDF cumulative as npqaud
	# tArrival = np.copy(occupancy)  # iniitalize tArrival (? DO i need this??) as npquad
	# tArrival[:] = notYetArrived
	# tArrival[occupancy > 0] = 0
	# TODO: make listofTimes and listofNs not hard-coded in? make it a parameter and keep them as defaults
	listofTimes = [10,100,500,1000,5000,9999]  # should be 7 (theses + startT+1)
	listofNs = [100, 1000, 100000, 100000000, 1e10]  # should be 5 of these
	for t, occ in evolve2DLattice(occupancy, maxT, distribution, params, True ,boundary=absorbingBoundary):
		# if tArrival.shape[0] != occ.shape[0]:
		# 	# Note: this is fragile, we assume that doubling tArrival will always work
		# 	tArrival = doubleArray(tArrival, arraytype=tArrival.dtype, fillValue=notYetArrived)
		# tArrival[(occ > 0) & (np.isnan(tArrival))] = t #update tArrival array
		CDF += occ  # update CDF
		if t == startT+1:
			#TODO: now actually need to fix PDF contour so CDF
			# use expand dims so that vstacks stacksk in the 3rd dim (the time dim)
			stats = np.expand_dims(np.array([getContourRoughness(CDF, N) for N in listofNs]),axis=0)
		if t in listofTimes:
			# p, a, r, d, d2, 1/N,
			tempstats = np.expand_dims(np.array([getContourRoughness(CDF, N) for N in listofNs]),axis=0)
			stats = np.vstack((stats,tempstats))
	# return tArrival.astype(np.quad), occ, CDF, np.array(stats)
	# however occ doesn't need to be cast to quad b/c in evolve2DLattice it's returned in executeMoves as quads
	# this is necesary b/c we want to accumulate quad precision as we evolve PDFs
	return occ, CDF, np.array(stats)
	# to plot with logs, you have to do like:
	# plt.imshow(np.log(PDF,where=(PDF!=0),out=np.zeros_like(PDF)).astype(float))
	# otherwise you get inf issues or graphic issues


def evolveAgents(occupancy, maxT, distribution, params, startT=1,
							 absorbingRadius = None):
	"""
	Evolves a 2DLattice with dynamic scaling.
	:param occupancy: initial occupancy, can be a number (NParticles) or an existing array
	:param maxT: timestep you want to go out to (integer)
	:param distribution: string; specify distribution from which biases pulled
	:param params: the parameters of the specified distribution
	:param startT: optional arg, starts at 1
	:param absorbingRadius: optional parameter to set location of absorbing boundary condition
	:return occ: the final evolved occupancy array
	:return tArrival: the array with the time of first arrival for every site in the occupancy array
	"""
	notYetArrived = np.nan #-1
	if distribution == 'dirichlet':
		params = float(params)
	else:
		params = None
	L = (maxT-1)//2 #if maxT = 10000, then this should be 4999
	temp = np.arange(-L, L + 1)
	x, y = np.meshgrid(temp, temp)
	distToOrigin = np.sqrt(x**2 + y**2)
	# if absorbingRadius is None:
	# 	absorbingRadius = L - 1
	# absorbingBoundary = (distToOrigin <= absorbingRadius)
	if np.isscalar(occupancy):
		# TOOD: why does evolveAgents break if I start with an occupancy that isn't [[1e10]] ?
		occupancy = np.array([[occupancy]], dtype=float)  # if given a scalar (ie NParticles or something), initializes array
		# occupancy = np.zeros((2 * L + 1, 2 * L + 1))
		# occupancy[L, L] = 1
	# initialize the array to throw in the time of first arrivals
	tArrival = np.copy(occupancy) #this is a copy of occ so it should also have npquad type
	tArrival[:] = notYetArrived
	tArrival[occupancy > 0] = 0
	# TODO: make listofTimes and listofNs not hard-coded in? make it a parameter and keep them as defaults
	listofTimes = [10,100,500,1000,5000,9999]
	for t, occ in evolve2DLattice(occupancy, maxT, distribution, params, False, boundary=True):
		if tArrival.shape[0] != occ.shape[0]:
			# Note: this is fragile, we assume that doubling tArrival will always work
			tArrival = doubleArray(tArrival, arraytype=tArrival.dtype, fillValue=notYetArrived)
		tArrival[(occ > 0) & (np.isnan(tArrival))] = t #update tArrival array
		if t == startT+1:
			stats = np.array(getTArrivalRoughness(tArrival,t)) #p, a, r, d, d2, t
		if t in listofTimes:
			tempstats = np.array(getTArrivalRoughness(tArrival,t))
			stats = np.vstack((stats,tempstats))
	# need to cast tArrival to quads here b/c it doesn't really matter when I cast it
	# however occ doesn't need to be cast to quad b/c in evolve2DLattice it's returned in executeMoves as quads
	# this is necesary b/c we want to accumulate quad precision as we evolve PDFs
	return tArrival.astype(np.quad), occ, np.array(stats)

# wrapper for evolve2DLattice
def run2dAgent(occupancy, maxT, distribution, params, PDF):
	for t, occ in evolve2DLattice(occupancy, maxT, distribution, params, PDF):
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
	notYetArrived = np.nan
	# initialize the moments & mask
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
			goodData = np.invert(np.isnan(tArrival)) # (tArrival != notYetArrived)
		# if you are somewhere in the middle of the list, first check that the array sizes will be the same
		# or alternatively make them the same
		else:
			if tArrMom1.shape[0] < tArrival.shape[0]:
				# if not the same size, change the moment arrays to be the same size as the incoming tArrival array
				tArrMom1 = changeArraySize(tArrMom1, tArrival.shape[0], fillval=notYetArrived)
				tArrMom2 = changeArraySize(tArrMom2, tArrival.shape[0], fillval=notYetArrived)
				goodData = changeArraySize(goodData, tArrival.shape[0], fillval=False)
			if tArrMom1.shape[0] > tArrival.shape[0]:
				# if not the same size, change the moment arrays to be the same size
				# as the incoming tArrival array
				tArrival = changeArraySize(tArrival, tArrMom1.shape[0], notYetArrived)
			# now cumulatively add the moments
			goodData *= np.invert(np.isnan(tArrival)) # (tArrival != notYetArrived)
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

# 26 April --> break getPerimeterAreaTau into two parts: 1) getting the binary image, and
#    2) getting the measurement for that specific state (new function)
def getStateRoughness(binaryImage): # change name
	""" function that takes in a binary image (bools) & calcs perimeter area roughness etc
	:param binaryImage: 2d binary image (tArrival  or occ or something)
	:return: perimeter, area, roughness, radius moment1, radius moment2: instantaneous measurements
		of the inputted state (binary image)
	"""
	# find  edge of the binary image
	edge = (binaryImage ^ m.binary_erosion(binaryImage))
	# get radius of each point from origin
	i, j = np.where(edge)
	if len(i) == 0:
		print("no i's found on edge")
	L = binaryImage.shape[0] // 2
	i, j = i - L, j - L
	radius, theta = cartToPolar(i, j)
	# get moments of radius
	radiusMoment1 = np.mean(radius)
	radiusMoment2 = np.mean(np.square(radius))
	# calculate the roughness of the state
	perimeter = np.sum(edge)
	area = np.sum(binaryImage)
	if area == 0:
		print("Area 0!")
	roughness = perimeter/np.sqrt(area)
	# return perimeter, area, roughness, etc of the state
	return perimeter, area, roughness, radiusMoment1,radiusMoment2

#7 April - once I put getStateRoughness in the generator loop then this function is obsolete?
# or I rewrite this to just do mask + getStateRoughness; leave the for i loop out of it??
def getTArrivalRoughness(tArrival, tau):
	""" Calculate the surface roughness of tArrivals as a function of time by (#edge pixels)/(total pixels reached)^1/2
	Takes one tArrival array and roughness stats. as function of time tau
	:param tArrival: the tArrival array, going out to its max time
	:return instantaneousStats: np array with (perimeter, area, roughness, radius moment 1, radius moment2, tau
	"""
	notYetArrived = np.nan
	# initialize the instantaneousStats w/ the stats at tau=1
	mask = (tArrival <= tau) & (np.invert(np.isnan(tArrival)))
	instantaneousStats = getStateRoughness(mask)+(tau,) #p, a, r, d, d2, tau
	# loop through the rest of the taus to build up the stats array ACTUALLY DONT DO THIS
	# for i in range(2, np.max(tArrival[np.invert(np.isnan(tArrival))].astype(int))+1):
	#     tempMask = (tArrival <= i) & (np.invert(np.isnan(tArrival)))
	#     instantaneousStats = np.vstack((instantaneousStats, getStateRoughness(tempMask)+(i,))) #p, a, r, d, d2, t
	#     # print("instantaneousStats shape:", instantaneousStats.shape)
	return instantaneousStats #p, a, r, d, d2, t

# This one.... I think goes inside the generator loop
def getContourRoughness(Array, N):
	''' Calculate roughness of an evolved CDF as a function of 1/NParticles by
	looking at contour of probability >= 1/N. implicitly also a function of time
	goes inside the evolve2Dlattice generator loop
	:param Array: 2D array
	:param N: the number of particles for which you want probability to be >= 1/N
	:return: perimeter, area, roughness, radius moment 1, adius moment 2, N
	'''
	#TODO: I only need to roll when doing PDF...
	#rolledPDF =  Array + np.roll(Array,1) # because checkerboard, binary erosion won't work
	mask = (Array >= 1/N) #binary image
	#TODO: check if binary image bad? UPDATE NO THIS ISNT THE WAY TO CHECK
	if np.all(mask == False):
		print("Empty binary image")
		print(f"N = {N} and max val = ",np.max(Array))
	roughnessStats = getStateRoughness(mask) #get roughness vals for that specific 1/N
	return roughnessStats +(1/N,) #p, a, r, d, d2, 1/N


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
	firstStats = np.load(f'{path}/{filelist[0]}').astype(float)
	mean = firstStats
	moment2 = firstStats ** 2
	# cumulatively calculate mean, second moment
	for file in filelist[1:]:
		tempStats = np.load(f'{path}/{file}').astype(float)
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
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, True, float, boundary=boundary):
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
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, True, float, boundary=boundary):
		# Get probabilities inside sphere
		if t in ts: 
			Rs = list(np.array(vs * t).astype(int))
			
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
	
	Example
	-------
	tMax = 100
	L = 250
	R = L-1
	Rs = [5, 10]
	linefile = 'Line.txt'
	distribution = 'dirichlet'
	params = 1/10
	measureOnSphere(tMax, L, R, Rs, distribution, params, linefile)
	'''
	ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
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

	# Create occupancy array
	occ = np.zeros((2 * L + 1, 2 * L + 1))
	occ[L, L] = 1
	
	x = np.arange(-L, L+1)
	xx, yy = np.meshgrid(x, x)
	dist_to_center = np.sqrt(xx ** 2 + yy ** 2)
	boundary = dist_to_center <= R
	
	# Need to make sure occ doesn't change size
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, 
								  True, boundary=boundary):
		# Get probabilities inside sphere
		if t in ts:
			Rs = np.array(vs * t).astype(int)
			# You might want to flip things to get the tail cdf, 
			# but it shouldn't matter
			xCDF = np.flip(np.cumsum(np.sum(occ, axis=1)))
			
			indeces = Rs + L
			indeces_outside_array = indeces >= len(xCDF)
			indeces[indeces_outside_array] = 0

			probs = xCDF[indeces]
			probs[indeces_outside_array] = 0
			
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
	
	x = np.arange(-L, L+1)
	xx, yy = np.meshgrid(x, x)
	dist_to_center = np.sqrt(xx ** 2 + yy ** 2)
	boundary = dist_to_center <= R

	ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))
	# Need to make sure occ doesn't change size
	for t, occ in evolve2DLattice(occ, tMax, distribution, params, True, float, boundary=boundary):
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




# old functions i'm saving until i can delete them
# 2 May 2024 --> have taken getPerimeterAreaTau and getRoughness and combined into getTArrivalRoughness
# so this function is now useless
# 26 April --> Need to rename... or maybe this can now go inside a general "get tArrivalRoughness"
# def getPerimeterAreaTau(tArrivalArray, tau):
#     """
#     Goes inside getRoughness. Finds Roughness parameters for a single tArrival array at specific time tau
#     :param tArrivalArray: array of tArrivals
#     :param tau: the time at which you want to measure roughness (int)
#     :return instantaneousStats: roughness stats. for a tArrival array at specific tau
#     :return tau: also returns the time at which youre looking for roughness stats
#     """
#     notYetArrived = np.nan
#     # turn tArrival array into a binary image based on specified value of tau
#     mask = (tArrivalArray <= tau) & (np.invert(np.isnan(tArrivalArray))) # (tArrivalArray > notYetArrived)) # get binary image specific to tArrival
#     # get the measurements for that state
#     instantaneousStats = getStateRoughness(mask)
#     return (instantaneousStats + (tau,)) #p, a, r, radiusmoment1, radiusmoment2, time
#
# # 2 May 2024 --> have taken getPerimeterAreaTau and getRoughness and combined into getTArrivalRoughness
# # so this function is now useless
# # 9 April getRoughnessNew --> getRoughness
# def getRoughness(tArrivalArray):
#     """
#     Calculate the surface roughness of tArrivals as a function of time by
#     (#edge pixels)/(total pixels reached)^1/2
#     If perfectly smooth, min roughness is like 2*(pi)^1/2.
#     Takes one tArrival array and roughness stats. as function of tau
#     :param tArrivalArray: the tArrival array, going out to max time tau
#     :return stats: numpy array of shape (tau, 5);
#         stats[:,0] returns perimeter, [:,1] area, [:,2]  roughness
#         [:,3]  avg. dist to edge, [:,4] returns (avg dist to edge)^2,
#         [:5] returns tau
#     """
#     # initialize stats with the tau = 1 (since startT = 1 in generateFirstArrivals)
#     stats = getPerimeterAreaTau(tArrivalArray, 1)  # p,a,r,d,d2,t
#     # add the rest of the stats using vstack
#     for i in range(2, np.max(tArrivalArray[np.invert(np.isnan(tArrivalArray))].astype(int))+1):
#         tempstats = getPerimeterAreaTau(tArrivalArray, i)
#         stats = np.vstack((stats, tempstats))
#     return stats


