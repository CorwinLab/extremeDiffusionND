import numpy as np
import os
import glob
import h5py
import json

# moments calculation for files saved as .h5
def getStatsh5py(path,tCutOff=None,takeLog=True):
    """
    Calculates mean, variance, etc. of ln[Probability outside sphere]
    Parameters
    ----------
    path: str,  something like "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3/L1000/tMax2000"
    tCutOff: optional, integer. time at which you want to cut off data before taking stats
    takeLog: boolean, default true.

    Returns
    -------
    saves Stats.h5 to the path given as a parameter

    """
    with open(f"{path}/variables.json",'r') as v:
        variables = json.load(v)
    time = np.array(variables['ts'])
    # TODO: fix this once I push the other thing, to fix the range issue
    # maxTime = time[-2]
    maxTime = time[-1] -1  # because of the range issue?
    print(maxTime)
    files = glob.glob(f"{path}/*.h5")
    # print(f"files: {files}")
    # TODO: add string option for if takeLog=False
    statsFile = h5py.File(f"{path}/Stats.h5",'a')
    moments = ['mean','secondMoment','var','skew']
    with h5py.File(files[0], 'r') as f:
        for regime in f['regimes'].keys():
            statsFile.require_group(regime)
            for moment in moments:
                statsFile[regime].require_dataset(moment, shape=f['regimes'][regime].shape, dtype=float)
                statsFile[regime][moment][:] = np.zeros(f['regimes'][regime].shape, dtype=float)
    num_files = 0
    for file in files:
        # print(file)
        with h5py.File(file, 'r') as f:
            if f.attrs['currentOccupancyTime'] < maxTime:
                print(f"file: {file}")
                print("Skipping")
                continue
            for regime in f['regimes'].keys():
                probs = f['regimes'][regime][:].astype(float)
                # TODO: figure out how to print something if this throws the error
                assert np.sum(np.isnan(probs)) == 0

                statsFile[regime]['mean'][:] += np.log(probs)
                statsFile[regime]['secondMoment'][:] += np.log(probs) ** 2
                # debugging prints
                # print(f"file: {file} \n regime: {regime}")
                # print(f"mean (unnormalized): {statsFile[regime]['mean'][:]}")
            num_files += 1
    for regime in statsFile.keys():
        statsFile[regime]['mean'][:] /= num_files
        statsFile[regime]['secondMoment'][:] /= num_files
        statsFile[regime]['var'][:] = statsFile[regime]['secondMoment'][:] - statsFile[regime]['mean'][:] ** 2


# moments calculation for files saved as .npy from the old code
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
    files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f))) and (not f.startswith('.'))]
    if 'info.npz' in files:
        files.remove('info.npz')
    if 'stats.npz' in files:
        files.remove('stats.npz')
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
        # print(f"file: {file}")
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
    if not takeLog:
        temp = "statsNoLog.npz"
    else:
        temp = "stats.npz"
    np.savez_compressed(os.path.join(path, temp), mean=moment1, variance=variance,
                        skew=skew, excessKurtosis=kurtosis - 3)