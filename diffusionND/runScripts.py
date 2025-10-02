import numpy as np
import npquad
from .pyDiffusionND import DiffusionND
import sys 
import h5py
import os
import json
from datetime import date
from time import time as wallTime

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

	return np.full_like(time.astype(float),fill_value=1,dtype=float)

def sqrt(time):
	return np.sqrt(time)

def calculateRadii(times, velocity, scalingFunction):
	"""
	get list of radii = v*(function of time) for barrier for given times; returns array of (# times, # velocities)
	Ex: radii = calculateRadii(np.array([1,5,10]),np.array([[0.01,0.1,0.5]]),tOnLogT)
	To get the original velocities, call radiiVT[0,:]
	"""

	funcVals = scalingFunction(times)
	funcVals = np.expand_dims(funcVals, 1)
	return velocity * funcVals

def getListOfTimes(maxT, startT=1, num=500):
	"""
	Generate a list of times, with approx. 10 times per decade (via np.geomspace), out to time maxT
	:param maxT: the maximum time to which lattice is being evolved
	:param startT: initial time at which lattice evolution is started
	:param num: number of times you want
	:return: the list of times
	"""
	return np.unique(np.geomspace(startT, maxT, num=num).astype(int))

def evolveAndMeasurePDF(ts, mostRecentTime, tMax, Diff, saveFileName, saveOccupancyFileName):
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
	hours = 3
	seconds = hours * 3600

	while Diff.time < tMax:
		Diff.iterateTimestep()
		if Diff.time in ts:
			# First need to pull out radii at current time we want
			idx = list(ts).index(Diff.time)
			radiiAtTimeT = []

			with h5py.File(saveFileName, 'r+') as saveFile:
				# Change because radii are now attached to the save file
				for regimeName in saveFile['regimes'].keys():
					radii = saveFile['regimes'][regimeName].attrs['radii']
					regimeRadiiAtTimeT = radii[idx, :]
					radiiAtTimeT.append(regimeRadiiAtTimeT)
				
				# Shape of resulting array is (regimes, velocities)
				radiiAtTimeT = np.vstack(radiiAtTimeT)
				probs = Diff.integratedProbability(radiiAtTimeT)
				
				# Now save data to file
				for count, regimeName in enumerate(saveFile['regimes'].keys()):
					saveFile['regimes'][regimeName][idx, :] = probs[count, :]
				
				# For ease, we will save the occupancy every time we 
				# write data to the file
				saveFile.attrs['currentOccupancyTime'] = Diff.time
				Diff.saveOccupancy(saveOccupancyFileName)
				startTime = wallTime()
				
		# Save at final time and if haven't saved for XX hours
		# Because it might take longer than 3 hours to go between
		# save times
		if (wallTime() - startTime >= seconds) or (Diff.time == tMax-1):
			# Save current time and occupancy to make restartable
			with h5py.File(saveFileName, 'r+') as saveFile:
				saveFile.attrs['currentOccupancyTime'] = Diff.time
				Diff.saveOccupancy(saveOccupancyFileName)
			
			# Reset the timer
			startTime = wallTime()

def runDirichlet(L, ts, velocities, params, directory, systID):
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
	ts = np.array(ts)
	velocities = np.array(velocities)
	tMax = max(ts)

	saveFileName = os.path.join(directory, f"{str(systID)}.h5")
	saveOccupancyFileName = os.path.join(directory, f"Occupancy{systID}.bin")

	with h5py.File(saveFileName, 'a') as saveFile: 
		# Define the regimes we want to study
		#regimes = [linear, np.sqrt, tOnSqrtLogT]
		regimes = [linear]
		# Check if "regimes" group has been made and create otherwise
		if 'regimes' not in saveFile.keys():
			saveFile.create_group("regimes")

			for regime in regimes: 
				saveFile['regimes'].create_dataset(regime.__name__, shape=(len(ts), len(velocities)), track_order=True)
				saveFile['regimes'][regime.__name__].attrs['radii'] = calculateRadii(ts, velocities, regime)

		# Load save if occupancy is already saved
		# Eric says the following should be a function.
		if ('currentOccupancyTime' in saveFile.attrs.keys()) and (os.path.exists(saveOccupancyFileName)):
			mostRecentTime = saveFile.attrs['currentOccupancyTime']
			print(f"Loaded file from time {mostRecentTime}", flush=True)
			Diff = DiffusionND.fromOccupancy(params, L, saveOccupancyFileName, mostRecentTime)
		else:
			# Otherwise, initialize as normal
			Diff = DiffusionND(params, L)
			saveFile.attrs['currentOccupancyTime'] = 0
			mostRecentTime = 0

	# actually run and save data
	evolveAndMeasurePDF(ts, mostRecentTime, tMax, Diff, saveFileName, saveOccupancyFileName)

	# To save space we delete the occupancy when done
	os.remove(saveOccupancyFileName)

def getExpVarX(distName, params):
	'''
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
		ExpX += np.sum(xvals * rand_vals) ** 2
		
	ExpX /= num_samples

	return ExpX

def saveVars(vars, save_file):
	"""
	Save experiment variables to a file along with date it was ran
	"""
	for key, item in vars.items():
		if isinstance(item, np.ndarray):
			vars[key] = item.tolist()
	
	with open(save_file, "w+") as file:
		json.dump(vars, file)
