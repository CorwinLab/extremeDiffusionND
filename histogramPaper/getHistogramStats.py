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
    # first file, should be (t by # radii)
    # note that w/ past line/point, the # radii is sectioned into 3 parts
    # [:, :50] sqrt, [:,50:100] critical, [:,100:] linear
    if measurement == 'line':
        firstFile = np.load(os.path.join(path,"FinalPoint0.npy"))
        finalProbsFileName = os.path.join(savePath, "FinalProbs"+"Line"+".npy")
        statsFileName = os.path.join(savePath, "Line"+fileName)
    elif measurement == 'point':  # point
        firstFile = np.load(os.path.join(path, "FinalPoint0.npy"))
        # "
        finalProbsFileName = os.path.join(savePath, "FinalProbs"+"Point"+".npy")
        statsFileName = os.path.join(savePath, "Point"+fileName)
    else:  #circle, should be named w/o ayny weird naming
        firstFile = np.load(os.path.join(path,"Final0.npy"))
        finalProbsFileName = os.path.join(savePath, "FinalProbs.npy")
        statsFileName = os.path.join(savePath, fileName)
    print(f"filenames: \n {finalProbsFileName} \n {statsFileName}")
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
