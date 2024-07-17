import evolve2DLattice as ev
import sys
import os
import numpy as np
import argparse as ap

# OLD VERSION
def runDataAndAnalysis(directory, sysID, occupancy, MaxT, distribution, params, PDF, absorbingradius):
    path = f"{directory}"
    if PDF:  # to better label directories
        path = path + "PDF"
    else:
        path = path + "Agents"
    if not os.path.exists(path):  # check if exists already, create if doesn't
        os.mkdir(path)
        print(f"{path} has been created.")
    if PDF:  # if evolving PDF then use evolvePDF, save the relevant stuff
        pdf, integratedPDF, pdfStats, integratedPDFStats, time, boundary = ev.evolvePDF(MaxT, distribution,
                                                                    params, startT=1,
                                                                    absorbingRadius = absorbingradius)
        np.savez_compressed(f"{path}/{sysID}.npz", pdf=pdf, integratedPDF=integratedPDF, pdfStats=pdfStats,
                            integratedPDFStats=integratedPDFStats, time=time, absorbingBoundary=boundary)
    else:  # if evolving agents then use evolveAgents, save relevant stuff
        tArrival, occ, tArrStats, boundary, times  = ev.evolveAgents(occupancy, MaxT, distribution,
                                                   params, startT=1, absorbingRadius=absorbingradius)
        np.savez_compressed(f"{path}/{sysID}.npz",tArrival = tArrival, occupancy = occ,
                            tArrivalStats = tArrStats, absorbingBoundary=boundary,times=times)


if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: runDataAndAnalysis.py <directoryName> <system ID> <occupancy> "
              "<maxT> <distribtuion> <params> <PDF> <number of systems> <absorbingRadius>")
        sys.exit(1)
    directoryName = sys.argv[1]  # String
    sysID = int(sys.argv[2])  # Integer
    occupancy = int(float(sys.argv[3]))  # Integer, cast from float to allow for scientific notation
    MaxT = int(sys.argv[4])  # Integer
    distribution = str(sys.argv[5])  # string for distribution name
    params = sys.argv[6]  # i think this needs to be a list
    PDF = eval(str(sys.argv[7]))  # Bool
    numSystems = int(sys.argv[8])  # Integer (why do I have this..)
    for i in range(0,10):
        print(f"sys.argv{i}= {sys.argv[i]}, type = {type(sys.argv[i])}")
    if sys.argv[9] == 'off':  # this is maybe a dumb way to correctly cast the absorbing radius parameter
        absorbingradius = str(sys.argv[9])
    elif sys.argv[9] == 'None':
        absorbingradius = None
    else:
        absorbingradius = int(sys.argv[9])
    for i in range(numSystems):
        runDataAndAnalysis(directoryName, sysID + i, occupancy, MaxT, distribution, params, PDF, absorbingradius)
