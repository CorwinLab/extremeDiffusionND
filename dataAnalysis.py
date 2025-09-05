import numpy as np
import os
import glob
import h5py
import json
from memEfficientEvolve2DLattice import getExpVarXDotProduct
from randNumberGeneration import getRandomDistribution

# moments calculation for files saved as .h5
def getStatsh5py(path,takeLog=True):
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
    with open(f"{path}/variables.json",'r') as v:
        variables = json.load(v)
    time = np.array(variables['ts'])
    maxTime = time[-1] -1  # because of the range issue?
    print(maxTime)

    files = glob.glob(f"{path}/*.h5")
    if not takeLog:  # if you want stats for P
        print("taking stats for P")
        if f"{path}/Stats.h5" in files:
            print('ignoring existing Stats.h5')
            files.remove(f"{path}/Stats.h5")  # ignore existing stats file for lnP (assumes done first)
        if f"{path}/StatsNoLog.h5" in files:
            print('rewriting StatsNoLog.h5')
            files.remove(f"{path}/StatsNoLog.h5")
        fileName = "StatsNoLog.h5"  #
    else:  # if you want stats for lnP (default)
        if f"{path}/Stats.h5" in files:
            print('overwriting existing Stats.h5')
            files.remove(f"{path}/Stats.h5")  # ignore existing stats file for lnP (assumes done first)
        if f"{path}/StatsNoLog.h5" in files:
            print('ignoring StatsNoLog.h5')
            files.remove(f"{path}/StatsNoLog.h5")
        print('taking stats for lnP')
        fileName = "Stats.h5"
    if f"{path}/FinalProbs.h5" in files:  # ignore final probability files
        files.remove(f"{path}/FinalProbs.h5")

    statsFile = h5py.File(f"{path}/{fileName}",'w')
    moments = ['mean', 'secondMoment', 'var', 'thirdMoment', 'skew']
    # initialize the analysis file with array of 0s with the correct shape
    with h5py.File(files[0], 'r') as f:
        for regime in f['regimes'].keys():
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
                    with np.errstate(divide='ignore'):  # prevent it from getting mad about divide by 0
                        if takeLog:  # stats for ln(P)
                            statsFile[regime]['mean'][:] += (np.log(probs)).astype(np.float64)
                            statsFile[regime]['secondMoment'][:] += (np.log(probs) ** 2).astype(np.float64)
                            statsFile[regime]['thirdMoment'][:] += (np.log(probs) ** 3).astype(np.float64)
                        else:  # stats for P
                            statsFile[regime]['mean'][:] += (probs).astype(np.float64)
                            statsFile[regime]['secondMoment'][:] += (probs ** 2).astype(np.float64)
                            statsFile[regime]['thridMoment'][:] += (probs ** 3).astype(np.float64)
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
                                            3*statsFile[regime]['mean'][:]*sigma**2
                                            - statsFile[regime]['mean'][:]** 3) / (sigma**3)
    statsFile.close()

# the following functions are for the Universal Fluctuations paper!
# inherently dependent on the way we save data but thats ok
# this shouldn't  change on if we're taking var[lnP] vs var[P]
def processStats(statsFile):
    """
    takes a h5 file with the saved, calculated stats of systems
    and outputs an array of [var(lnP), radius, time, lambda_ext, mean[lnP], mean[P],var[P] ]
    that array should then get thrown into the calculation of s(r,t,lambda_ext)
    params: file path to an stats file generated by getStatsh5py
    returns:
        data: np array where 0th axis gives varLnP, radii, t, lambda_ext, meanLnP, meanP, varP
    """
    stats = h5py.File(f"{statsFile}","r")
    # setup for grabbing list of vs and ts
    filePath = os.path.split(statsFile)[0]  # returns the directory data & stats in
    regimes = ['linear','tOnSqrtLogT','sqrt']
    testFile = h5py.File(f"{filePath}/0.h5","r")
    with open (f"{filePath}/variables.json","r") as v:
        variables = json.load(v)
    ts = np.array(variables['ts'])  # in order from 0 to 9999
    vs = np.array(variables['velocities'])
    # calculation of lambda_ext
    distName = variables['distName']
    params = np.array(variables['params'])
    if '' in params:
        params = ''
    # get list of Var_nu[E^xi[Y]]
    expVarX = getExpVarXDotProduct(distName, params)
    # turn into D_ext and D
    D_ext = (1/2) * expVarX  # D_ext includes factor of 1/2 by defn.
    D = 1/2  # by defn includes factor of 1/2
    # reshaping of arrays
    longTs = np.tile(ts,vs.shape[0])  # turn ts array into 1d array of size (336 * 221)

    # calculate lambda_ext
    # note that lambda_ext also includes factor of 1/2 by defn.
    # turn lambda_ext into 1d array of size (336*21 = 7056)
    lambda_ext = np.array([(1/2) * (D_ext / (D - D_ext))]*(ts.shape[0]*vs.shape[0]))

    # fence post problem
    firstRadii = testFile['regimes'][regimes[0]].attrs['radii'].flatten('F')
    linearVarFirst = stats[regimes[0]]['var'][:].flatten('F')
    meanFirst = stats[regimes[0]]['mean'][:].flatten('F')
    data = np.array([linearVarFirst, firstRadii, longTs, lambda_ext, meanFirst])
    if distName == 'Dirichlet':
        label = distName + str(params[0])
    else:
        label = distName
    for regime in regimes[1:]:
        # (336*21 = 7056 1d array) of linear radii
        # flatten in column major so its like (v[0] radii for all t, v[1] radii for all t, ... etc)
        radii = testFile['regimes'][regime].attrs['radii'].flatten(order='F')
        # returns (331*21 = 7056 1d array)
        var = stats[regime]['var'][:].flatten(order='F')
        mean = stats[regime]['mean'][:].flatten(order='F')
        tempData = np.array([var, radii, longTs, lambda_ext, mean])
        data = np.hstack((data, tempData))
    # indices for data: (6, t.shape * v.shape)
    # the 0th index: 0 = var[lnP or P], 1 = radii, 2 = t, 3 = lambda, 4 = mean[ln P or P]
    # note that if you want to pull out a specific val, such as tMax
    # indices = np.where(data[2,:] == tmax)
    return data, label

# indpt of varLnP vs varP
def masterCurveValue(radii, times, lambda_ext):
    """
    note that if you want specific vals for some tMax you do like
    vals = masterCurveValue(data[1,:][indices],data[2,:][indices],data[3,:][indices])
    plt.loglog(vals,data[0,:][indices],'.') or something
    Parameters
    ----------
    radius: list of radii
    times: list of times
    lambda_ext: list of lambda_ext
    Returns
    -------
    f(r(t),t,lambda_ext) = (lambda_ext) r^2 / t^2
    """

    scalingFunction = lambda_ext * (radii**2 / times**2)
    return scalingFunction


def getListOfLambdas(statsList):
    # var_nu[E^xi[Y]]
    expVarXList = []
    # needs to be defined as 1/2 * (D_ext / (1 - D_ext) )
    lambdaList = []
    for path in statsList:
        filePath = os.path.split(path)[0]  # returns the directory data & stats in
        with open(f"{filePath}/variables.json", "r") as v:
            variables = json.load(v)
        # calculation of lambda_ext
        distName = variables['distName']
        params = np.array(variables['params'])
        if '' in params:
            params = ''
        expVarX = getExpVarXDotProduct(distName, params)
        expVarXList.append(expVarX)
        D_ext = (1 / 2) * expVarX  # D_ext includes factor of 1/2 by defn.
        D = 1 / 2  # by defn includes factor of 1/2
        lambda_ext = (1 / 2) * (D_ext / (D - D_ext))
        lambdaList.append(lambda_ext)
    return np.array(expVarXList), np.array(lambdaList)

# vlpMax depends on if we look at var[lnP] in which case vlpMax ~ 10^-3.. idk for varP tho
def prepLossFunc(statsList, tMaxList, vlpMax, alpha=1, rMin=3):
    """
    takes list of stats files, chops off values where var[lnP] > vlpMax
    and then computes
    std( g - t^alpha ) where g = vlp / ( lambda*v**2 )
    """
    # initialize arrays for concatenate
    scalingFuncs = np.array([])
    vars = np.array([])
    vs = np.array([])
    times = np.array([])
    ls = np.array([])
    for i in range(len(statsList)):
        print(statsList[i])
        file = statsList[i]
        tempData, label = processStats(file)
        for j in range(len(tMaxList)):
            # grab times we're interested in, and mask out the small radii (r<1) vals.
            # chop data above vlpMax
            if vlpMax is not None:
                indices = np.array(np.where((tempData[2, :] == tMaxList[j])
                                        & (tempData[1, :] >= rMin)
                                        & (tempData[0,:] < vlpMax))).flatten()
            else:
                indices = np.array(np.where((tempData[2, :] == tMaxList[j])
                                        & (tempData[1, :] >= rMin))).flatten()
            # pull out the r, t, and lambdas of our masked data, then calc lambda r^2/t^2
            r = tempData[1, indices]  # radii
            t = tempData[2, indices]  # time
            l = tempData[3, indices]  # lambda_ext
            scalingFuncVals = masterCurveValue(r, t, l)
            # cast our velocities, assuming r = v t^alpha
            vLin = r / t**alpha

            # smash into list
            scalingFuncs = np.concatenate((scalingFuncs, scalingFuncVals))
            vars = np.concatenate((vars, tempData[0,indices]))
            vs = np.concatenate((vs, vLin))
            times = np.concatenate((times, t))
            ls = np.concatenate((ls, l))
    return scalingFuncs, vars, vs, times, ls
