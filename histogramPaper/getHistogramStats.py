import glob
import numpy as np
import npquad
import h5py
from tqdm import tqdm
import os
import json

def calcStatsForHistogram(path, takeLog=True):
    """
       Calculates mean, second moment, variance. of ln[Probability outside sphere]' or just of prob outside sphere
       Also saves probabilities at some tFinal of every system into 1 file
       Parameters
       ----------
       path: str,  something like "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3/L1000/tMax2000"
           should be the path to the directory in which your data is contained
       takeLog, boolean: if True (default) takes stats of ln(P); if false, takes stats of P
       Returns
       -------
       saves Stats.h5 to the path given as a parameter

       """
    with open(f"{path}/variables.json", 'r') as v:
        variables = json.load(v)
    time = np.array(variables['ts'])
    maxTime = time[-1] - 1  # because of the range issue?
    print(maxTime)

    files = glob.glob(f"{path}/*.h5")
    ignoreThese = ['Stats.h5','StatsNoLog.h5','FinalProbs.h5']
    for temp in ignoreThese:  # ignore existing stats/nolog/finalprob files
        if f"{path}/{temp}" in files:
            files.remove(f"{path}/{temp}")
    if takeLog:
        fileName = "Stats.h5"
    else:
        fileName = "StatsNoLog.h5"
    # initialize files
    finalProbsFile = h5py.File(os.path.join(path, "FinalProbs.h5"),'a')
    statsFile = h5py.File(os.path.join(path,fileName), 'a')

    moments = ['mean', 'secondMoment', 'var', 'thirdMoment', 'skew']
    # initialize the analysis file with array of 0s with the correct shape
    with h5py.File(files[0], 'r') as f:
        for regime in f['regimes'].keys():
            # TODO: initialize finalProbsFile
            statsFile.require_group(regime)
            for moment in moments:
                statsFile[regime].require_dataset(moment, shape=f['regimes'][regime].shape, dtype=np.float64)
                statsFile[regime][moment][:] = np.zeros(f['regimes'][regime].shape, dtype=np.float64)
    # start the calculation
    num_files = 0
    for file in files:
        try:
            with h5py.File(file, 'r') as f:
                if f.attrs['currentOccupancyTime'] < maxTime:
                    print(f"Skipping file: {file}, current occ. is {f.attrs['currentOccupancyTime']}")
                    continue
                for regime in f['regimes'].keys():
                    probs = f['regimes'][regime][:].astype(float)
                    # if np.sum(np.isnan(probs)) == 0 is false, then throws an error
                    assert np.sum(np.isnan(probs)) == 0
                    if takeLog:
                        temp = np.log(probs)
                    else:
                        temp = probs
                    with np.errstate(divide='ignore'):  # prevent it from getting mad about divide by 0
                        statsFile[regime]['mean'][:] += (temp).astype(np.float64)
                        statsFile[regime]['secondMoment'][:] += (temp ** 2).astype(np.float64)
                        statsFile[regime]['thirdMoment'][:] += (temp ** 3).astype(np.float64)
                num_files += 1
        except Exception as e:  # skip file if corrupted, also say its corrupted
            print(f"{file} is corrupted!")
            continue
    statsFile.attrs['numberOfFiles'] = num_files
    for regime in statsFile.keys():  # normalize, then calc. var and skew
        statsFile[regime]['mean'][:] /= num_files
        statsFile[regime]['secondMoment'][:] /= num_files
        statsFile[regime]['var'][:] = statsFile[regime]['secondMoment'][:] - statsFile[regime]['mean'][:] ** 2

        with np.errstate(invalid='ignore', divide='ignore'):
            statsFile[regime]['var'][:] = statsFile[regime]['secondMoment'][:] - statsFile[regime]['mean'][:] ** 2
            sigma = np.sqrt(statsFile[regime]['var'][:])

            statsFile[regime]['skew'][:] = (statsFile[regime]['thirdMoment'][:] -
                                            3 * statsFile[regime]['mean'][:] * sigma ** 2
                                            - statsFile[regime]['mean'][:] ** 3) / (sigma ** 3)
    statsFile.close()

if __name__ == "__main__":
    dir = '/mnt/talapasData/data/Dirichlet/ALPHA1/L1000/1000/'

    meanVarFile = h5py.File(os.path.join(dir, "MeanVar.h5"), 'a')
    finalProbs = h5py.File(os.path.join(dir, "FinalProbs.h5"), 'a')

    fileNums = 5_000_000

    maxTime = 999
    moments = ['mean', 'secondMoment', 'thirdMoment', 'var', 'skew']

    firstFileName = os.path.join(dir, "0.h5")
    with h5py.File(firstFileName, 'r') as f:
        for regime in f['regimes'].keys():
            meanVarFile.require_group(regime)
            finalProbs.require_dataset(f'temp{regime}', shape=(fileNums, f['regimes'][regime].shape[1]), dtype=float)

            for moment in moments:
                meanVarFile[regime].require_dataset(moment, shape=f['regimes'][regime].shape, dtype=float)
                meanVarFile[regime][moment][:] = np.zeros(f['regimes'][regime].shape, dtype=float)

    num_files = 0
    for fileID in tqdm(range(fileNums)):
        fileName = os.path.join(dir, f'{fileID}.h5')
        try:
            with h5py.File(fileName, 'r') as f:
                if f.attrs['currentOccupancyTime'] < maxTime:
                    continue

                for regime in f['regimes'].keys():
                    probs = f['regimes'][regime][:]

                    finalProbs[f'temp{regime}'][num_files, :] = np.log(probs[-1, :]).astype(float)

                    with np.errstate(divide='ignore'):
                        meanVarFile[regime]['mean'][:] += (np.log(probs)).astype(float)
                        meanVarFile[regime]['secondMoment'][:] += (np.log(probs) ** 2).astype(float)
                        meanVarFile[regime]['thirdMoment'][:] += (np.log(probs) ** 3).astype(float)

                num_files += 1
        except Exception as e:
            continue

    meanVarFile.attrs['numberOfFiles'] = num_files
    print(num_files)

    for regime in meanVarFile.keys():
        meanVarFile[regime]['mean'][:] /= num_files
        meanVarFile[regime]['secondMoment'][:] /= num_files
        meanVarFile[regime]['thirdMoment'][:] /= num_files

        with np.errstate(invalid='ignore', divide='ignore'):
            meanVarFile[regime]['var'][:] = meanVarFile[regime]['secondMoment'][:] - meanVarFile[regime]['mean'][:] ** 2
            sigma = np.sqrt(meanVarFile[regime]['var'][:])
            meanVarFile[regime]['skew'][:] = (meanVarFile[regime]['thirdMoment'][:] - 3 * meanVarFile[regime]['mean'][:] * sigma ** 2 - meanVarFile[regime]['mean'][:] ** 3) / sigma ** 3

        nonzeroProbs = finalProbs[f'temp{regime}'][:num_files, :]
        finalProbs.create_dataset(regime, data=nonzeroProbs)
        del finalProbs[f'temp{regime}']

    meanVarFile.close()
    finalProbs.close()