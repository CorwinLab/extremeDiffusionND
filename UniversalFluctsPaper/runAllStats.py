import dataAnalysis as d

if __name__ == "__main__":
    path003 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.03162278/L5000/tMax10000"
    path01 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.1/L5000/tMax10000"
    path03 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA0.31622777/L5000/tMax10000"
    path1 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA1/L5000/tMax10000"
    path3 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA3.1622776/L5000/tMax10000"
    path10 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA10/L5000/tMax10000"
    path31 = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/dirichlet/ALPHA31.622776/L5000/tMax10000"
    pathLogNormal = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/logNormal/0,1/L5000/tMax10000"
    pathDelta = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/Delta/L5000/tMax10000"
    pathCorner = "/mnt/talapasData/data/memoryEfficientMeasurements/h5data/Corner/L5000/tMax10000"
    fullList = [path003, path01, path03, path1, path3, path10, path31, pathLogNormal, pathDelta, pathCorner]

    for path in fullList:
        print(f"running new stats for {path}")
        d.getStatsh5py(path,takeLog=False)

    print("done!")