import evolve2DLattice as ev
import sys
import os
import numpy as np

# 8 April changing all dirichlet --> distribution and getRoughness to getRoughnessNew... add params
# 9 April getRoughnessNew --> getRoughness
def runDataAndAnalysis(directory, sysID, occupancy, MaxT, distribution, params, PDF):
    # check if path exists or create if doesnt; also create ANOTHER directory to throw statistics into
    path = f"{directory}"
    statsPath = f"{directory}"+"Statistics"
    if PDF:  # to better label directories
        path = path + "PDF"
        statsPath = path + "Statistics"
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(statsPath)
        print(f"{path} and {statsPath} have been created.")

    # then generate the tArrival and occupancy array and save it
    tArrival, occ = ev.generateFirstArrivalTime(occupancy, MaxT, distribution, params, PDF)
    # save the tArrival and occ to the directory we created with the systID as the filename
    np.savez_compressed(f"{path}/{sysID}.npz", tArrival=tArrival, occ=occ)

    # #now I want to do the analysis...
    statistics = ev.getRoughness(tArrival)  # use getRoughnessNew instead
    np.save(f"{statsPath}/{sysID}.npy", statistics)  # since outputs one array, use np.save and .npy


if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: runDataAndAnalysis.py <directoryName> <system ID> <occupancy> "
              "<maxT> <distribtuion> <params> <PDF> <number of systems>")
        sys.exit(1)
    directoryName = sys.argv[1]  # String
    sysID = int(sys.argv[2])  # Integer
    occupancy = int(float(sys.argv[3]))  # Integer, cast from float to allow for scientific notation
    MaxT = int(sys.argv[4])  # Integer
    # dirichlet = eval(sys.argv[5])  # Bool
    distribution = str(sys.argv[5])  # string for distribution name
    params = sys.argv[6]  # i think this needs to be a list
    PDF = eval(sys.argv[7])  # Bool
    numSystems = int(sys.argv[8])  # Integer

    for i in range(numSystems):
        runDataAndAnalysis(directoryName, sysID + i, occupancy, MaxT, distribution, params, PDF)
