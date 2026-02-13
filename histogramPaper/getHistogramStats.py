import numpy as np
import npquad
import h5py
from tqdm import tqdm
import os
import json
import sys

# TODO: at some point fix this to deal with my stupid filename convention
# for past a line & past a point. because i'm STUPID.
### rewriting for new format
def calculateStatistics(path, savePath, takeLog=True, lookAtNum=None,measurement='circle'):
    """
    procedure. calculates mean, 2nd moment, variance, 3rd moment, and kurtosis (?) of ln[Prob(meas)]
    or, if takeLog=False, just the moments of Prob(meas).saves to a stats file.
    also saves ln(probs) at some tFinal of every system into 1 file (this is because we don't
    want to have to open every 50k file twice.
    Parameters
    ----------
    path: str, "data/linePointMeas/1000/Line/" or something. path to where the data lives
    savePath: str, "data/linePointMeas/1000/Line/" or something. Saves Stats.npy
    takeLog: boolean, default true. If false, calcs stats of Prob(meas) instead of ln(Prob(meas))
    lookAtNum: int, number of files from 0 to whatever. to look at

    """
    os.makedirs(savePath, exist_ok=True)
    expected_file_num = 50000  # eventaully i will have 50k systems for histograms
    with open(f"{path}/variables.json", 'r') as v:
        variables = json.load(v)
    tMax = variables['tMax']
    # times = np.unique(np.geomspace(1,tMax,500).astype(int))
    if takeLog:
        fileName = "Stats.npy"
    else:
        fileName = "StatsNoLog.npy"
    # initialize stats files
    finalProbsFileName = os.path.join(savePath,"FinalProbs.npy")
    statsFileName = os.path.join(savePath, fileName)

    # first file, should be (t by # radii)
    # note that w/ past line/point, the # radii is sectioned into 3 parts
    # [:, :50] sqrt, [:,50:100] critical, [:,100:] linear
    if measurement == 'line':
        firstFile = np.load(os.path.join(path,"FinalPoint0.npy"))
    elif measurement == 'point':  # point
        firstFile = np.load(os.path.join(path, "FinalPoint0.npy"))
    else:  #circle
        firstFile = np.load(os.path.join(path,"Final0.npy"))
    # initialize finalProbs as (#files, # radii)
    finalProbs = np.zeros(shape=(expected_file_num,firstFile.shape[1]))
    moment1, moment2, moment3, moment4 = firstFile, firstFile*firstFile, np.power(firstFile,3), np.power(firstFile,4)
    if lookAtNum is not None:  # only look at 0-lookAtNum (ie a subset)
        maxID = lookAtNum
    else:
        maxID = expected_file_num
    num_files = 0
    n_corrupted = 0
    for fileID in tqdm(range(maxID)):
        # try to open the file in a try/except. if good, add to the moments
        try:
            # handle the stupid fucking filenaming. why am i stupid.
            if measurement == 'line':
                logProbs = np.load(os.path.join(path, f"FinalPoint{fileID}.npy"))
            elif measurement == 'point':  # point
                logProbs = np.load(os.path.join(path, f"FinalPoint{fileID}.npy"))
            else:  # circle
                logProbs = np.load(os.path.join(path, f"Final{fileID}.npy"))
            # update moments. prevent it from getting mad about divide by 0
            with np.errstate(divide='ignore'):  # we are going to += the shit out of this
                moment1 += logProbs
                moment2 += logProbs * logProbs
                moment3 += np.power(logProbs, 3)
                moment4 += np.power(logProbs, 4)
            num_files += 1
            # update the final probability list
            finalProbs[num_files, :] = logProbs[-1, :]
        except Exception as e:  # skip file if corrupted, also say its corrupted
            print(f"{fileID} is corrupted!")
            n_corrupted += 1
            continue
    # normalize the moments and calc. mean var skew kurtosis
    moment1 /= num_files
    moment2 /= num_files
    moment3 /= num_files
    moment4 /= num_files
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = moment1
        variance = moment2 - np.square(moment1)
        skew = (moment3 - 3*moment1*variance - np.power(moment1,3)) / (variance**(3/2))
        kurtosis = (moment4 - 4*moment1*moment3 + 6*(moment1**2)*moment2 - 3*np.power(moment1,4))/np.square(variance)
    # save files
    stats = np.array([mean,variance,skew,kurtosis])
    np.save(statsFileName, stats)
    # save final probs to file
    nonzeroProbs = finalProbs[:num_files,:]  # chop off the part of the array we didn't use, if any
    # savve final probs file using finalProbsFileName
    np.save(finalProbsFileName, nonzeroProbs)
    return

# will save the probabilities at some tFinal of every system into 1 file






# OLD SINCE WE DONT USE H5 NOW
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
    expected_file_num = 50000  # want to overshoot here even if its not actually
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

def SMA(data, windowsize):
    """ returns the simple moving avg. of data"""
    i = 0
    movingAvg = []
    while i < len(data) - windowsize + 1:
            wA = np.nansum(data[i:i+windowsize]) / windowsize
            movingAvg.append(wA)
            i += 1
    return np.array(movingAvg)

def getTStr(t):
    if t < 10:
        tStr = f"00{t}"
    elif 10 <= t < 100:
        tStr = f"0{t}"
    else:
        tStr = f"{t}"
    return tStr

# TODO: stop hardcoding in the regimes... but for now its fine
def finalProbsAtTs(path, savePath, tList,lookAtNum=None):
    """
       Saves probabilities at some tFinal of every system into 1 file
       Parameters
       ----------
       path: str,  something like "data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3/L1000/tMax2000"
           should be the path to the directory in which your data is contained
       takeLog, boolean: if True (default) takes stats of ln(P); if false, takes stats of P
       lookAtNum: int, number of files from 0 to whatever to look at
       Returns
       -------
       saves t{t}Probs.h5 to the given path (file full of ln(LAST probability measurement)
       """
    os.makedirs(savePath,exist_ok=True)
    expected_file_num = 5000000  # want to overshoot here even if its not actually
    with open(f"{path}/variables.json", 'r') as v:
        variables = json.load(v)
    time = np.array(variables['ts'])
    maxTime = time[-1]
    print(maxTime)
    # initialize files
    finalProbsFile = h5py.File(os.path.join(savePath, "FinalProbs.h5"),'a')
    if lookAtNum is not None:  # only look at 0-lookAtNum (ie a subset)
        maxID = lookAtNum
    else:
        maxID = expected_file_num
    # initialize the analysis file with array of 0s with the correct shape
    firstFile = os.path.join(path,"0.h5")
    regimes = ['sqrt','tOnSqrtLogT','linear']
    with h5py.File(firstFile, 'r') as f:
        for t in tList:
            tStr = getTStr(t)
            tGrp = finalProbsFile.create_group(f"t{tStr}")
            for regime in f['regimes'].keys():
                tGrp.require_dataset(f'temp{regime}',shape=(maxID,f['regimes'][regime].shape[1]), dtype=np.float64)
    # start the calculation
    num_files = 0
    n_corrupted = 0
    for fileID in tqdm(range(maxID)):
        file = os.path.join(path, f'{fileID}.h5')
        try:
            with h5py.File(file, 'r') as f:
                if f.attrs['currentOccupancyTime'] < maxTime:
                    print(f"Skipping file: {file}, current occ time. is {f.attrs['currentOccupancyTime']}")
                    continue
                for t in tList:
                    idx = list(time).index(t)  # pull index of the time in which ur interested in
                    tStr = getTStr(t)
                    for regime in regimes:
                        probs = f['regimes'][regime][:]
                        finalProbsFile[f"t{tStr}"][f'temp{regime}'][num_files,:] = np.log(probs[idx, :]).astype(np.float64)
                        assert np.sum(np.isnan(probs)) == 0
                num_files += 1
        except Exception as e:  # skip file if corrupted, also say its corrupted
            print(f"{fileID} is corrupted!")
            # print(e)
            n_corrupted += 1
            if n_corrupted % 1000 == 0:
                print(f"corrupted: {n_corrupted}, good: {num_files}")
            continue
    finalProbsFile.attrs['numberOfFiles'] = num_files
    for t in tList:
        tStr = getTStr(t)
        for regime in regimes:
            # cut off the allocated array to exclude the corrupted files (which should be all 0s or nans)
            nonzeroProbs = finalProbsFile[f"t{tStr}"][f'temp{regime}'][:num_files, :]
            finalProbsFile[f"t{tStr}"].create_dataset(regime, data=nonzeroProbs)
            del finalProbsFile[f"t{tStr}"][f'temp{regime}']
    print("no. corrupted files: ", n_corrupted)
    print("no. files: ", num_files)
    finalProbsFile.close()


if __name__ == "__main__":
    # Test Code. assumes always taking log and always looking at the full no. of files
    # dataDirectory, savePathDirectory
    dataDirectory = sys.argv[1]
    savePathDirectory = sys.argv[2]

    # iterates through all the files in dataDirectory and writes statsfile in SavePathDirectory
    calcStatsForHistogram(dataDirectory, savePathDirectory)
