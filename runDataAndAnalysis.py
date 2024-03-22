import evolve2DLattice as ev
import sys
import os
import numpy as np

def runDataAndAnalysis(directoryName,systID,occupancy,MaxT,dirichlet):
    #assume working in extremeDiffusionND
    #ok first check if path exists or create if doesnt; stay in extremeDiffusionND tho
    #also want to create ANOTHER directory to throw statistics into
    path = f"{directoryName}"
    statsPath = f"{directoryName}"+"Statistics"
    # create a folder to throw all runs into, or check that it exists
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(statsPath)
        #os.chdir(path)
        print(f"{directoryName} and {statsPath} have been created.")
    #then generate the tArrival and occupancy array and save it
    tArrival, occ = ev.generateFirstArrivalTimeAgent(occupancy, MaxT,dirichlet)
    #save the tArrival and occ to the directory we created with the systID as the filename
    np.savez_compressed(f"{path}/{systID}.npz", tArrival=tArrival, occ=occ)

    #now I want to do the analysis...
    perimeter, area, time, roughness, avgDist, avgDist2 = ev.getRoughness(tArrival)
    #and save the analysis quantities to statspath/systid.npz
    np.savez_compressed(f"{statsPath}/{systID}.npz",perimeter = perimeter, area = area, time = time, roughness = roughness,avgBoundaryDist = avgDist,avgBoundaryDist2 = avgDist2)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: script.py <directoryName> <system ID> <occupancy> <maxT>")
        sys.exit(1)
    directoryName = sys.argv[1]
    systID = int(sys.argv[2])
    occupancy = int(float(sys.argv[3]))
    MaxT=int(sys.argv[4])
    numSystems = int(sys.argv[5])
    dirichlet = sys.argv[6]

    for i in range(numSystems):
        runDataAndAnalysis(directoryName,systID + i, occupancy,MaxT,dirichlet)