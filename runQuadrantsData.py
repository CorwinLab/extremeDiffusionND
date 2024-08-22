import evolve2DLattice as ev
import os
import numpy as np
import argparse as ap
import ast

# base directory should be /projects/jamming/fransces/data/quadrants/
def runQuadrantsData(baseDirectory, sysID, tMax, L, R, vs, distribution, params):
    '''run file to generate cumulative probability in quadrants
    :param baseDirectory: str, path to which you save data
    :param sysID: int
    :param tMax: maximum time to evolve system to
    :param L: int, dist. from center of occupancy array
    :param R: int, should be L-1, determins radius of absorbing boundary
    :param vs: np array, list of "velocities", about a decade between'''
    #TODO: figure out data saving structure because my brain is stalling
    os.makedirs(baseDirectory, exist_ok=True)
    boxSaveFile = os.path.join(baseDirectory,'Box'+str(sysID)+'.txt')
    hLineSaveFile = os.path.join(baseDirectory, 'hLine'+str(sysID)+'.txt')
    vLineSaveFile = os.path.join(baseDirectory,'vLine'+str(sysID)+'.txt')
    sphereSaveFile = os.path.join(baseDirectory,'sphere'+str(sysID)+'.txt')

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
    # parser.add_argument('--absorbingRadius', type=int, default=False,
    #                     help='specify the radius of absorbing boundary, if <0 then no boundary, if not specified then uses default scaling')
    parser.add_argument('sysID', type=int, help='system ID passed in; should be the slurm array number')
    parser.add_argument("L", type=int, help="specify dist. from origin to edge of occupancy")
    parser.add_argument("--vs", help="specify list of velocities, so quadrants move w/ v*t^(1/2)")
    parser.add_argument("--R",type=int, default=None, help="specify radius of absorbing boundary, default L-1")
    parser.add_argument('--params', help='specify the parameters of distribution (only dirichlet for now, 1/10)',
                        default=None)
    args = parser.parse_args()
    # workaround to specify R = L-1 in case you don't put in a radius
    if args.R is None:
        args.R = args.L - 1

    # argparse dumb so hard code in equally spaced velocities
    # the min. velocity to check is the one where at t=tMax, the radius
    # or moving line has moved exactly 1
    # the max is when, at t=tmax, v sqrt(t) has crossed t exaclty once. 
    vs = np.geomspace(1/np.sqrt(args.tMax),np.sqrt(args.tMax), 21)

    # call it once, instead of numSys
    runQuadrantsData(args.baseDirectory, args.sysID, args.tMax, args.L, args.R, vs,
                     args.distribution, args.params)