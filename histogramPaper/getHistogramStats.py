import numpy as np
import npquad
import h5py
from tqdm import tqdm
import os
import json
import sys

def calcStatsForHistogram(path, savePath, takeLog=True,lookAtNum=None):
    """
       Calculates mean, second moment, variance. of ln[Probability outside sphere]' or just of prob outside sphere
       Also saves probabilities at some tFinal of every system into 1 file
       Parameters
       ----------
       path: str,  something like "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3/L1000/tMax2000"
           should be the path to the directory in which your data is contained
       takeLog, boolean: if True (default) takes stats of ln(P); if false, takes stats of P
       lookAtNum: int, number of files from 0 to whatever to look at
       Returns
       -------
       saves Stats.h5 to the path given as a parameter
       saves finalProbs.h5 to the given path (file full of ln(LAST probability measurement)
       """
    os.makedirs(savePath,exist_ok=True)
    expected_file_num = 5000000  # want to overshoot here even if its not actually
    with open(f"{path}/variables.json", 'r') as v:
        variables = json.load(v)
    time = np.array(variables['ts'])
    maxTime = time[-1]  # because of the range issue?
    print(maxTime)
    if takeLog:
        fileName = "Stats.h5"
    else:
        fileName = "StatsNoLog.h5"
    # initialize files
    finalProbsFile = h5py.File(os.path.join(savePath, "FinalProbs.h5"),'a')
    statsFile = h5py.File(os.path.join(savePath,fileName), 'a')

    moments = ['mean', 'secondMoment', 'var', 'thirdMoment', 'skew']
    # initialize the analysis file with array of 0s with the correct shape
    firstFile = os.path.join(path,"0.h5")
    with h5py.File(firstFile, 'r') as f:
        for regime in f['regimes'].keys():
            statsFile.require_group(regime)
            finalProbsFile.require_dataset(f'temp{regime}',shape=(expected_file_num,f['regimes'][regime].shape[1]), dtype=np.float64)
            for moment in moments:
                statsFile[regime].require_dataset(moment, shape=f['regimes'][regime].shape, dtype=np.float64)
                statsFile[regime][moment][:] = np.zeros(f['regimes'][regime].shape, dtype=np.float64)
    # start the calculation
    if lookAtNum is not None:  # only look at 0-lookAtNum (ie a subset)
        maxID = lookAtNum
    else:
        maxID = expected_file_num
    num_files = 0
    n_corrupted = 0
    for fileID in tqdm(range(maxID)):
        file = os.path.join(path, f'{fileID}.h5')
        try:
            with h5py.File(file, 'r') as f:
                if f.attrs['currentOccupancyTime'] < maxTime:
                    print(f"Skipping file: {file}, current occ. is {f.attrs['currentOccupancyTime']}")
                    continue
                for regime in f['regimes'].keys():
                    probs = f['regimes'][regime][:]
                    finalProbsFile[f'temp{regime}'][num_files,:] = np.log(probs[-1, :]).astype(np.float64)
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
            print(f"{fileID} is corrupted!")
            n_corrupted += 1
            if n_corrupted % 1000 == 0:
                print(f"corrupted: {n_corrupted}, good: {num_files}")
            continue
    statsFile.attrs['numberOfFiles'] = num_files
    for regime in statsFile.keys():  # normalize, then calc. var and skew
        statsFile[regime]['mean'][:] /= num_files
        statsFile[regime]['secondMoment'][:] /= num_files
        statsFile[regime]['thirdMoment'][:] /= num_files
        statsFile[regime]['var'][:] = statsFile[regime]['secondMoment'][:] - statsFile[regime]['mean'][:] ** 2

        with np.errstate(invalid='ignore', divide='ignore'):
            statsFile[regime]['var'][:] = statsFile[regime]['secondMoment'][:] - statsFile[regime]['mean'][:] ** 2
            sigma = np.sqrt(statsFile[regime]['var'][:])
            statsFile[regime]['skew'][:] = (statsFile[regime]['thirdMoment'][:] -
                                            3 * statsFile[regime]['mean'][:] * sigma ** 2
                                            - statsFile[regime]['mean'][:] ** 3) / (sigma ** 3)
        nonzeroProbs = finalProbsFile[f'temp{regime}'][:num_files,:]
        finalProbsFile.create_dataset(regime, data=nonzeroProbs)
        del finalProbsFile[f'temp{regime}']
    print("no. corrupted files: ", n_corrupted)
    print("no. files: ", num_files)
    statsFile.close()
    finalProbsFile.close()

if __name__ == "__main__":
    # Test Code. assumes always taking log and always looking at the full no. of files
    # dataDirectory, savePathDirectory
    dataDirectory = sys.argv[1]
    savePathDirectory = sys.argv[2]

    # iterates through all the files in dataDirectory and writes statsfile in SavePathDirectory
    calcStatsForHistogram(dataDirectory, savePathDirectory)
