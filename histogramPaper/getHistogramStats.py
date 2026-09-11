import numpy as np
from tqdm import tqdm
import os
import json
from scipy.special import logsumexp as LSE
# from directedPolymer import logSumExp

#TODO: this function works and when doing like
# testmoment1 = firstFile
# for array in arrays, testMoment1 = momentLogP(array, testMoment1, 1), it works
# so somehow the way I implemented it in calcExpectationLogPAndLogExpecationP is screwed up
# maybe the difference between np.power and logP*logP is part of the issue?
def momentLogP(newLogP, currentMomentVal, nthMoment):
    # when calculating the moments of logP, since I have logP stored
    # i ccan just do updatedMoment = currentMomentVal + np.power(newLogP,nthMoment)
    # updatedMoment = currentMomentVal + np.power(newLogP, nthMoment)
    if nthMoment == 1:
        updatedMoment = newLogP + currentMomentVal
    elif nthMoment == 2:
        updatedMoment = newLogP * newLogP + currentMomentVal
    else:
        updatedMoment = np.power(newLogP, nthMoment) + currentMomentVal
    return updatedMoment

# TODO: fix this please. i have no clue what's wrong with this.
def logMomentP(newLogP, currentLogMomentP, nthMoment):
    # calculating log(moments of P) is more difficult since I've stored logP, not P
    # to update this calculation I need to do
    # log(E[P^n]) = log(sum_i^N P^n / N) = log(sum_i^N exp(ln(P_i^n)) / N)
    # = log(sum_i^N exp(n lnP_i) )  - ln(N) (the lnN gets taken care of at the end)
    # = LSE(n*lnP)
    # scipy's LSE function axis=0 keeps the shape of the logP array
    updatedLogMoment = LSE([nthMoment*newLogP, currentLogMomentP],axis=0)
    return updatedLogMoment

#TODO: fix this because something is very wrong with this
# TODO: i don't get it though because i'm doing the exact same thing???
# TODO: OK i got the means (both logMean and meanLOG) to line up with the calculateStatistics and with NUMPY
# TODO: but now the variances won't line up
def calcExpectationLogPAndLogExpectationP(path, savePath, lookAtNum=None):
    """
    procedure. calcs mean[lnP], var[lnP], skew[lnP]. also calculates
    ln[MeanP], ln[VarP], ln[SkewP].
    To do the latter, implements a logsumexp() version of calculating the moments
    parameters. To do this we  need to initialize with the first file
    Also saves the logProbs at tMax
    path: string, which describes the path to where the data lives. this should include a variables.json file
    savePath: string, describes path where the stats files gets saved
    lookAtNum: int, number of files to look at (assumes starting from 0)
    """
    os.makedirs(savePath, exist_ok=True)
    expected_file_num = 50000
    statsLogPFileName = os.path.join(savePath,"statsLogP.npy")  # mean(logP), var(logP), etc...
    logStatsPFileName = os.path.join(savePath,"logStatsP.npy")  # log(MeanP), log(varP), ...
    finalLogProbsFileName = os.path.join(savePath,"FinalLogProbs.npy")
    print(f"filenames: \n {finalLogProbsFileName} \n {logStatsPFileName} \n {statsLogPFileName}")
    # initialize with the 1st file, which are logP values
    firstFile = np.load(os.path.join(path,"Final0.npy"))
    # moments of logP
    momentsOfLogP = np.array([firstFile,firstFile*firstFile, np.power(firstFile,3), np.power(firstFile,4)])
    # log(moments of P) need to be initialized with nthMoment*logP?
    logMomentsOfP = np.array([firstFile,2*firstFile, 3*firstFile, 4*firstFile])
    finalLogProbs = np.zeros(shape=(expected_file_num,firstFile.shape[1]))
    if lookAtNum is not None:  # only look at 0-lookAtNum (ie a subset)
        maxID = lookAtNum
    else:
        maxID = expected_file_num
    num_files = 1
    for fileID in tqdm(range(1,maxID)):  # already loaded in 0
        try:
            logProbs = np.load(os.path.join(path, f"Final{fileID}.npy"))
            with np.errstate(divide='ignore'):  # we are going to += the shit out of this
                for n in range(1, 5):  # iterate over the moments
                    momentsOfLogP[n-1,:,:] = momentLogP(logProbs, momentsOfLogP[n-1,:,:],n)
                    logMomentsOfP[n-1,:,:] = logMomentP(logProbs, logMomentsOfP[n-1,:,:],n)
            # update the final probability list
            finalLogProbs[num_files, :] = logProbs[-1, :]
            # now advance the counter of # of systems
            num_files += 1
        except Exception as e:  # skip file if corrupted, also say its corrupted
            print(f"{fileID} is corrupted or can't be opened!")
            continue
        # normalize the moments of logP and normalize the log(moments of P)
    momentsOfLogP /= num_files  # we normalize stats(logP) by dividing by N
    logMomentsOfP -= np.log(num_files)  # to normalzie log(statsP) we need to subtract off log(N)
    # use moments to calculate the stats
    with np.errstate(invalid='ignore', divide='ignore'):
        meanLogP = momentsOfLogP[0,:,:]
        varLogP = momentsOfLogP[1,:,:] - np.square(meanLogP)
        skewLogP = (momentsOfLogP[2,:,:] - 3 * meanLogP * varLogP - np.power(meanLogP, 3)) / (varLogP ** (3 / 2))
        kurtosisLogP = (momentsOfLogP[3,:,:] - 4 * meanLogP * momentsOfLogP[2,:,:] + 6 * (meanLogP ** 2) * momentsOfLogP[1,:,:] - 3 * np.power(meanLogP,4))/np.square(varLogP)
        # log(meanP), log(varP), log(skewP), etc
        logMeanP = logMomentsOfP[0,:,:]
        # for var, skew, etc. I need to use logsumexp again to avoid under/over flow
        # log(Var[P]) = log(E[P^2] - E[P]^2) = LSE(log(E[P^2]), -log[E[P]])^2 = LSE(log(E[P^2]), -2log(E[P]) ) ?
        logVarP = LSE([logMomentsOfP[1,:,:], 2*logMeanP],axis=0,b=[np.ones_like(logMeanP),np.full_like(logMeanP,-1)])
        # log(SkewP) = log( (nlM3 - 3*logVarP)/(logVarP^(3/2)) )
        # logSkewP = (moment3 - 3 * moment1 * variance - np.power(moment1, 3)) / (variance ** (3 / 2))
        # = LSE([log(E[P^3]), -((log3)+(logVarP)+(logmeanP)),-3log(P)],axis=0 ) - 3/2 log(VarP)
        logSkewP = (LSE([logMomentsOfP[2,:,:],(np.log(3)+logVarP+logMeanP),3*logMeanP],
                       axis=0,
                       b=[np.ones_like(logMeanP),np.full_like(logMeanP,-1),np.full_like(logMeanP,-1)])
                    - (3/2)*logVarP)
        logKurtosisP = (LSE([logMomentsOfP[3,:,:], (np.log(4)+logMomentsOfP[2,:,:]+logMeanP), (np.log(6)+2*logVarP+2*logMeanP),(np.log(3)+2*logMeanP)],
                            axis=0,
                            b=[np.ones_like(logMeanP),np.full_like(logMeanP,-1),np.ones_like(logMeanP),np.ones_like(logMeanP)])
                        - 2*logVarP)
    statsLogP = np.array([meanLogP, varLogP, skewLogP, kurtosisLogP])
    logStatsP = np.array([logMeanP, logVarP, logSkewP, logKurtosisP])
    nonzeroLogProbs = finalLogProbs[:num_files,:]  # chop off the part of the array we didn't use, if any
    np.save(statsLogPFileName, statsLogP)  # stats of logP
    np.save(logStatsPFileName, logStatsP)  # log(stats of P)
    np.save(finalLogProbsFileName, nonzeroLogProbs)  # final logProbs file
    print(f"finished! # completed files: {num_files}")
    return
#
# # for past a line & past a point. because i'm STUPID.
# TODO: NOTE THIS ONE MATCHES WHEN I USE NUMPY
def calculateStatistics(path, savePath, lookAtNum=None, measurement=None):
    """
    procedure. calculates mean, 2nd moment, variance, 3rd moment, and kurtosis (?) of ln[Prob(meas)]
    also saves ln(probs) at some tFinal of every system into 1 file (this is because we don't
    want to have to open every 50k file twice.
    Parameters
    ----------
    path: str, "data/linePointMeas/1000/Line/" or something. path to where the data lives
    savePath: str, "data/linePointMeas/1000/Line/" or something. Saves Stats.npy
    measurement: 'line' or 'point'; default is circle. tells the code which files to look for depending on which measurement
    lookAtNum: int, number of files from 0 to whatever. to look at

    """
    os.makedirs(savePath, exist_ok=True)
    expected_file_num = 50000  # eventaully i will have 50k systems for histograms
    # with open(f"{path}/variables.json", 'r') as v:
    #     variables = json.load(v)
    # tMax = variables['tMax']
    # times = np.unique(np.geomspace(1,tMax,500).astype(int))
    fileName = "Stats.npy"
    noLogFileName = "StatsNoLog.npy"
    # initialize stats files
    # first file, should be (t by # radii)
    # note that w/ past line/point, the # radii is sectioned into 3 parts
    # [:, :50] sqrt, [:,50:100] critical, [:,100:] linear
    if measurement == 'line':
        firstFile = np.load(os.path.join(path,"Final0.npy"))
        finalProbsFileName = os.path.join(savePath, "FinalProbs"+"Line"+".npy")
        statsFileName = os.path.join(savePath, "Line"+fileName)
        statsNoLogFileName = os.path.join(savePath,"Line"+noLogFileName)
    elif measurement == 'point':  # point
        firstFile = np.load(os.path.join(path, "FinalPoint0.npy"))
        # "
        finalProbsFileName = os.path.join(savePath, "FinalProbs"+"Point"+".npy")
        statsFileName = os.path.join(savePath, "Point"+fileName)
        statsNoLogFileName = os.path.join(savePath,"Point"+noLogFileName)
    else:  # circle
        # this one should only be the vt regime (60 velocities)
        firstFile = np.load(os.path.join(path,"Final0.npy"))
        finalProbsFileName = os.path.join(savePath, "FinalProbs"+"Circle"+".npy")
        statsFileName = os.path.join(savePath, "Circle"+fileName)
        statsNoLogFileName = os.path.join(savePath,"Circle"+noLogFileName)
    print(f"filenames: \n {finalProbsFileName} \n {statsFileName}")
    # initialize finalProbs as (#files, # radii)
    finalProbs = np.zeros(shape=(expected_file_num,firstFile.shape[1]))
    # with log
    moment1, moment2, moment3, moment4 = 0,0,0,0  # the fileID starts at 0 so its ok?
    # without log, need to initialize as logsumExp(first file)?
    # so I can't just initialize it with 0?
    m1NoLog, m2NoLog, m3NoLog, m4NoLog = 0,0,0,0
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
                logProbs = np.load(os.path.join(path, f"Final{fileID}.npy"))
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
            # update the final probability list
            finalProbs[num_files, :] = logProbs[-1, :]
            # now advance the counter
            num_files += 1
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
    print("# corrupted: ",n_corrupted)
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
