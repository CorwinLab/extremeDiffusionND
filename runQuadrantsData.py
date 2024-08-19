import evolve2DLattice as ev
import os
import numpy as np
import argparse as ap

# base directory should be /projects/jamming/fransces/data/quadrants/
def runQuadrantsData((baseDirectory, sysID, tMax, L, R, vs, distribution, params):
    #TODO: figure out data saving structure because my brain is stalling
    os.makedirs(baseDirectory, exist_ok=True)
    boxSaveFile = baseDirectory + sysID +'Box.txt'
    hLineSaveFile = baseDirectory + sysID +'hLine.txt'
    vLineSaveFile = baseDirectory + sysID + 'vLine.txt'
    sphereSaveFile = baseDirectory + sysID+'sphere.txt'

    # output: 4 files... but 1 directory?
    # ie give /projects/jamming/fransces/data/quadrants/
    ev.measureAtVsBox(tMax, L, R, vs, distribution, params,
                   boxSaveFile, hLineSaveFile, vLineSaveFile, sphereSaveFile)



if __name__ == "__main__":
    # initialize argparse
    parser = ap.ArgumentParser()
    parser.add_argument('baseDirectory', type=str, help=" specify directory to which data is saved")
    parser.add_argument('tMax', type=int, help='specify maximum time to which lattice is evolved')
    parser.add_argument('distribution', type=str, choices=['uniform', 'dirichlet', 'SSRW'],
                         help='specify "uniform", "dirichlet", or "SSRW" as the distribution from which biases are drawn')
    parser.add_argument('--params', help='specify the parameters of distribution (only dirichlet for now, 1/10)')
    # parser.add_argument('--absorbingRadius', type=int, default=False,
    #                     help='specify the radius of absorbing boundary, if <0 then no boundary, if not specified then uses default scaling')
    parser.add_argument('sysID', type=int, help='system ID passed in; should be the slurm array number')
    args = parser.parse_args()

    # call it once, instead of numSys
    runQuadrantsData(args.baseDirectory, args.sysID, args.tMax, args.distribution, args.params)