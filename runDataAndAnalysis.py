import evolve2DLattice as ev
# import sys
import os
import numpy as np
import argparse as ap

def runDataAndAnalysis(directory, sysID, occupancy, MaxT, distribution, params, PDF, absorbingradius):
    path = f"{directory}"
    # if PDF:  # to better label directories
    #     path = path + "PDF"
    # else:
    #     path = path + "Agents"
    # Make sure that the directory path exists after this line executes
    # This feels identical to wrapping it in an if statement but it
    # catches the potential race conditions
    os.makedirs(path, exist_ok=True)
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
    # initialize argparse
    parser = ap.ArgumentParser()
    parser.add_argument('directoryName', type=str, help=" specify directory to which data is saved")
    parser.add_argument('occupancy', type=float, help='specify initial occupancy of lattice')
    parser.add_argument('maxT', type=int, help='specify maximum time to which lattice is evolved')
    parser.add_argument('distribution', type=str, choices=['uniform', 'dirichlet', 'SSRW'],
                        help='specify "uniform", "dirichlet", or "SSRW" as the distribution from which biases are drawn')
    parser.add_argument('--params', help='specify the parameters of distribution (only dirichlet for now)')
    parser.add_argument('--isPDF', action='store_true', help='a boolean switch to turn pdf on')
    parser.add_argument('--isNotPDF', action='store_false', dest='isPDF', help='boolean switch to turn off pdf')
    parser.add_argument('--absorbingRadius', type=int, default=False,
                        help='specify the radius of absorbing boundary, if <0 then no boundary, if not specified then uses default scaling')
    parser.add_argument('sysID', type=int, help='system ID passed in; should be the slurm array number')
    args = parser.parse_args()

    # call it once, instead of numSys
    runDataAndAnalysis(args.directoryName, args.sysID, args.occupancy, args.maxT,
                           args.distribution, args.params, args.isPDF, args.absorbingRadius)