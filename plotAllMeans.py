import json
from matplotlib import pyplot as plt
import matplotlib
import dataAnalysis as d
import numpy as np
import matplotlib
from matplotlib.patches import FancyArrowPatch
from visualizeMeasurements import plotMean

if __name__ == "__main__":

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
    statsFileList = [path003, path01, path03, path1, path3, path10, path31,
                     pathLogNormal, pathDelta, pathCorner]

    with open("/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000/variables.json",
              "r") as v:
        variables = json.load(v)
    tMaxList = np.array(variables['ts'])

    # for dropping distributions which have very close values of lambda
    minimalStatsList = [path31, pathCorner, path1, pathDelta, path03, path01, path003]
    minExpVarXList, minLambdaList = d.getListOfLambdas(minimalStatsList)
    minmarkers = ['o'] + ['s'] + ['o'] + ['v'] + ['o'] * 3
    meanPath = "/home/fransces/Documents/Figures/Paper/2DRWREMean.png"
    plotMean(meanPath, minimalStatsList, tMaxList, markers=minmarkers, lambdaExtVals=minLambdaList)