import numpy as np
import npquad
from pyDiffusionND import DiffusionND
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

def evolveAndMeasurePDF(ts, tMax, Diff, saveFileName):
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
	
	for t in range(tMax):
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
				probs = Diff.integratedProbability(radiiAtTimeT)
				print(t, probs)
				
				# Now save data to file
				for count, regimeName in enumerate(saveFile['regimes'].keys()):
					saveFile['regimes'][regimeName][idx, :] = probs[count, :]
				
				# For ease, we will save the occupancy every time we 
				# write data to the file
				saveFile.attrs['currentOccupancyTime'] = t
				saveFile['currentOccupancy'][:] = Diff.PDF
				startTime = wallTime()
				
		hours = 3
		seconds = hours * 3600
		# Save at final time and if haven't saved for XX hours
		# Because it might take longer than 3 hours to go between
		# save times
		if (wallTime() - startTime >= seconds) or (t == tMax-1):
			# Save current time and occupancy to make restartable
			with h5py.File(saveFileName, 'r+') as saveFile:
				saveFile.attrs['currentOccupancyTime'] = t
				saveFile['currentOccupancy'][:] = Diff.PDF
			
			# Reset the timer
			startTime = wallTime()

		Diff.iterateTimestep()

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

	with h5py.File(saveFileName, 'a') as saveFile: 
		# Define the regimes we want to study
		regimes = [linear, np.sqrt, tOnSqrtLogT]
		# Check if "regimes" group has been made and create otherwise
		if 'regimes' not in saveFile.keys():
			saveFile.create_group("regimes")

			for regime in regimes: 
				saveFile['regimes'].create_dataset(regime.__name__, shape=(len(ts), len(velocities)))
				saveFile['regimes'][regime.__name__].attrs['radii'] = calculateRadii(ts, velocities, regime)

		# Load save if occupancy is already saved
		# Eric says the following should be a function.
		if ('currentOccupancyTime' in saveFile.attrs.keys()) and ('currentOccupancy' in saveFile.keys()):
			mostRecentTime = saveFile.attrs['currentOccupancyTime']
			occ = saveFile['currentOccupancy'][:]
			# Need to set occupancy and most recent time and params
		else:
			# Otherwise, initialize as normal
			Diff = DiffusionND(params, L)
			saveFile.create_dataset('currentOccupancy', data=Diff.PDF, compression='gzip')
			saveFile.attrs['currentOccupancyTime'] = 0

	# actually run and save data
	evolveAndMeasurePDF(ts, tMax, Diff, saveFileName)

	# To save space we delete the occupancy when done
	with h5py.File(saveFileName, 'r+') as saveFile:
		del saveFile['currentOccupancy']

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

if __name__ == "__main__":
	# Test Code
	# L, tMax, distName, params, directory, systID = 1000, 10000, 'Dirichlet', '1,1,1,1', './', 0

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

	ts = getListOfTimes(tMax - 1, 1)
	velocities = np.geomspace(10 ** (-3), 10, 21)

	vars = {'L': L, 
			'ts': ts,
			'velocities': velocities,
			'params': params,
			'directory': directory,
			'systID': systID}
	print(f"vars: {vars}")

	vars_file = os.path.join(directory, "variables.json")
	print(f"vars_file is {vars_file}")

	today = date.today()
	text_date = today.strftime("%b-%d-%Y")

	# Only save the variables file if on the first system
	if systID == 0:
		vars.update({"Date": text_date})
		saveVars(vars, vars_file)
		vars.pop("Date")

	start = wallTime()
	# runDirichlet(L, ts, velocities, params, directory, systID)
	runDirichlet(**vars)
	print(wallTime() - start, flush=True)