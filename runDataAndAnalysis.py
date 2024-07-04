import evolve2DLattice as ev
import sys
import os
import numpy as np

# 8 April changing all dirichlet --> distribution and getRoughness to getRoughnessNew... add params
# 9 April getRoughnessNew --> getRoughness
#TODO: fix to incorporate that roughness now computed w/in the loop
def runDataAndAnalysis(directory, sysID, occupancy, MaxT, distribution, params, PDF):
    # check if path exists or create if doesnt; also create ANOTHER directory to throw statistics into
    print("top of runDataAndAnalysis PDF", PDF, type(PDF))
    path = f"{directory}"
    statsPath = f"{directory}"+"Statistics"
    if PDF:  # to better label directories
        path = path + "PDF"
        # statsPath = path + "Statistics" (Don't need b/c stats in generator loop now)
    if not os.path.exists(path):
        os.mkdir(path)
        # os.mkdir(statsPath)
        print(f"{path} has been created.")

    if PDF:  # if evolving PDF then use evolvePDF, save the relevant stuff
        print("right before ev.evolvePDF", PDF, type(PDF))
        pdf, cdf, cdfStats = ev.evolvePDF(MaxT, distribution, params)
        # save the PDF, CDF, & CDF Stats to the directory with the systID as the filename
        np.savez_compressed(f"{path}/{sysID}.npz", pdf = pdf, cdf = cdf, cdfStats = cdfStats)
        #TODO: rewrite once i split generateFirstArrival
        # # now I want to do the analysis. on tArrival arrays (if agents)
        # if not PDF:
        #     statistics = ev.getRoughness(tArrival)
        #     # statistics = ev.getTArrivalRoughness(tArrival) # 29 Apr testing new functions for get Rougheness
        # else:  # if you are doing PDFs then we'll want the PDF roughness stats
        #     statistics = ev.getRoughness(occ)
        # np.save(f"{statsPath}/{sysID}.npy", statistics)  # since outputs one array, use np.save and .npy
    else:  # if evolving agents then use evolveAgents, save relevant stuff
        tArrival, occ, tArrStats = ev.evolveAgents(occupancy, MaxT, distribution, params)
        np.savez_compressed(f"{path}/{sysID}.npz",tArrival = tArrival, occupancy = occ, tArrivalStats = tArrStats)
        print(f"Saved {path}/{sysID}.npz")


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
    PDF = eval(str(sys.argv[7]))  # Bool
    numSystems = int(sys.argv[8])  # Integer (why do I have this..)

    for i in range(numSystems):
        runDataAndAnalysis(directoryName, sysID + i, occupancy, MaxT, distribution, params, PDF)
