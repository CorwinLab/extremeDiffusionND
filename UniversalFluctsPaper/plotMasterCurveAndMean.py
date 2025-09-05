import visualizeMeasurements as v
import numpy as np
import dataAnalysis as d
import json



if __name__ == "__main__":
    """
    generates mastercurve collapse fig and the mean collapse fig (fig 3 (bottom), and fig. 2) in paper
    """

    path003 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.03162278/L5000/tMax10000/Stats.h5"
    path01 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.1/L5000/tMax10000/Stats.h5"
    path03 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.31622777/L5000/tMax10000/Stats.h5"
    path1 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/Stats.h5"
    path3 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000/Stats.h5"
    path10 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000/Stats.h5"
    path31 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000/Stats.h5"
    pathLogNormal = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/logNormal/0,1/L5000/tMax10000/Stats.h5"
    pathDelta = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/Delta/L5000/tMax10000/Stats.h5"
    pathCorner = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000/Stats.h5"

    # for all data, all times
    fullList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]
    expVarXListFull, lambdaListFull = d.getListOfLambdas(fullList)
    fullMarkers = ['o'] * 7 + ['D'] + ['v'] + ['s']

    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json","r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])

    # # for dropping distributions which have very close values of lambda
    minSavePath = "/home/fransces/Documents/Figures/2DRWREMasterCurveReduced.pdf"
    minimalStatsList = [path31, pathCorner, path1, pathDelta, path03, path01, path003]
    minExpVarXList, minLambdaList = d.getListOfLambdas(minimalStatsList)
    minMarkers = ['o'] + ['s'] + ['o'] + ['v'] + ['o']*3
    v.plotMasterCurve(minSavePath, minimalStatsList, fullList, tMaxList,
                    minLambdaList, lambdaListFull, minMarkers, verticalLine=True)

    # makes the mean plot, but takes forever
    meanPath2 = "/home/fransces/Documents/Figures/2DRWREMeanFull.pdf"
    v.plotMean(meanPath2, fullList, tMaxList, markers=fullMarkers, lambdaExtVals=lambdaListFull)