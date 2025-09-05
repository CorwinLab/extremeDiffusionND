import sys 
import os
import numpy as np

#TODO!!: write in shit to ignore unfinished files
def getMeasurementMeanVarSkew(path, dist, tCutOff=None, takeLog=True):
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
    
    nFiles = 1
    lastvals = firstData[:, -1, :]
    lastvals = np.expand_dims(lastvals, axis=0)
    
    # load in rest of files to do mean var calc, excluding the 0th file
    for file in files[1:]:
        # print(f"file: {file}")
        data = np.load(f"{path}/{file}")
        if tCutOff is not None:  # only chop data if you give it a cutoff time
            data = data[:, :idx + 1, :]
        if takeLog:
            data = np.log(data)
        
        if (np.sum(np.isneginf(data[3, :, 0]))) > 1:
            continue
        lastvals = np.append(lastvals, np.expand_dims(data[:, -1, :], axis=0), axis=0)
        
        moment1 += data
        moment2 += np.square(data)
        moment3 += np.power(data, 3)
        moment4 += np.power(data, 4)
        nFiles += 1

    moment1 = moment1 / nFiles
    moment2 = moment2 / nFiles
    moment3 = moment3 / nFiles
    moment4 = moment4 / nFiles
    
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
    np.savez_compressed(os.path.join('./Data/', f'{dist}stats.npz'), mean=moment1, variance=variance,
                        skew=skew, excessKurtosis=kurtosis - 3)
    np.savez_compressed(os.path.join('./Data/', f'{dist}pdfVals.npz'), pdfVals=lastvals)

dir = '/mnt/talapasShared/'

# Folder is really just the name of the distribution
folders = os.listdir(dir)
folders = [f for f in folders if 'alpha' in f]

for f in folders:
    path = os.path.join(dir, f, 'L5000', 'tMax10000')

    getMeasurementMeanVarSkew(path, f, takeLog=True)