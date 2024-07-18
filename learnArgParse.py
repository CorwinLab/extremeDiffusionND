import evolve2DLattice as ev
import sys
import os
import numpy as np
import argparse as ap
def runDataAndAnalysisTEMP(directory, sysID, occupancy, MaxT, distribution, params, PDF, absorbingradius):
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
        tArrival, occ, tArrStats, boundary  = ev.evolveAgents(occupancy, MaxT, distribution,
                                                   params, startT=1, absorbingRadius=absorbingradius)
        np.savez_compressed(f"{path}/{sysID}.npz",tArrival = tArrival, occupancy = occ,
                            tArrivalStats = tArrStats, absorbingBoundary=boundary)

# from https://docs.python.org/3/howto/argparse.html
# https://docs.python.org/3/library/argparse.html
# https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments

if __name__ == "__main__":
    # initialize argparse
    parser = ap.ArgumentParser()
    parser.add_argument('directoryName', type=str,  help="required; specify directory to save data to")
    parser.add_argument('occupancy', type=float, help='required;specify initial occupancy of lattice')
    parser.add_argument('maxT', type=int,help='required; specify maximum time to which lattice is evolved')
    parser.add_argument('distribution',type=str, choices=['uniform','dirichlet','SSRW'],
                        help='required; specify "uniform", "dirichlet", "SSRW"')
    parser.add_argument('--params',help='optional; parameters of distribution')
    parser.add_argument('--isPDF',action='store_true',help='a boolean switch to turn pdf on')
    parser.add_argument('--isNotPDF', action='store_false', dest='isPDF', help='boolean switch to turn off pdf')
    # don't actually need numSystems because we're getting rid of the for loop for Talapas
    parser.add_argument('--numSystems',default=10,type=int,help='denote how many systems should be evolved')
    parser.add_argument('--absorbingRadius', type=int, default=False,
                        help='specify the radius of absorbing boundary, if <0 then no boundary, if not specified then defaults ')
    parser.add_argument('sysID', type=int, help='system ID passed in from slurm array?')
    args = parser.parse_args()

    # i think eventually we just want one thing?
    runDataAndAnalysisTEMP(args.directoryName, args.sysID,args.occupancy,args.maxT,
                           args.distribution,args.params,args.isPDF,args.absorbingRadius)

    #TODO: maybe do like.  if args.absorbingBoundary < 0 then don't return boundary??? idk
    # or maybe it's fine? since the data still generates and the issue is with loading it back in
    # for i in range(args.numSystems):
    #    tempSysID = i
    #    runDataAndAnalysisTEMP(args.directoryName, i, (args.occupancy), args.maxT,
    #                           args.distribution, args.params, args.isPDF, args.absorbingRadius)

    # TODO: implement sysID?
    # runDataAndAnalysis(args.directoryName, sysID + i, int(args.occupancy), args.maxT, args.distribution, args.params, args.isPDF, args.absorbingradius)



    # # directoryName = sys.argv[1]  # String
    # # sysID = int(sys.argv[2])  # Integer
    # # occupancy = int(float(sys.argv[3]))  # Integer, cast from float to allow for scientific notation
    # # MaxT = int(sys.argv[4])  # Integer
    # # distribution = str(sys.argv[5])  # string for distribution name
    # # params = sys.argv[6]  # i think this needs to be a list
    # # PDF = eval(str(sys.argv[7]))  # Booltr
    # # numSystems = int(sys.argv[8])  # Integer (why do I have this..)
    # for i in range(0,10):
    #     print(f"sys.argv{i}= {sys.argv[i]}, type = {type(sys.argv[i])}")
    # if sys.argv[9] == 'off':  # this is maybe a dumb way to correctly cast the absorbing radius parameter
    #     absorbingradius = str(sys.argv[9])
    # elif sys.argv[9] == 'None':
    #     absorbingradius = None
    # else:
    #     absorbingradius = int(sys.argv[9])
    # # for i in range(numSystems):
    # #     runDataAndAnalysis(directoryName, sysID + i, occupancy, MaxT, distribution, params, PDF, absorbingradius)
    # for i in range(numSystems):
    #     print("")










#TODO: delete this once familiar with argparse

# initialize argparse
# import argparse as ap
#parser = ap.ArgumentParser()

# # introduce positional arguments
# # specify that "echo" is an allowed command-line option
# # specifying "help" will tell the user what it does
# parser.add_argument("echo", help="echo the string you use here")
# # this method returns some data from options specified, in this case echo
# # the variable is some form of 'magic' that argparse performs for free (no need to specify
# #  which variable that val. is stored in).
# args = parser.parse_args()
# print(args.echo)


# more compilcated code
# parser.add_argument('square', help='display a square of a given number',
#                     type=int)
# args = parser.parse_args()
# # only works bc we've specified int as the type
# print(args.square**2)

# # introduce optional arguments
# parser.add_argument("--verbose", help='increase output verbosity',
#                     action='store_true')
# # action='store_true' tells it that not specifying a value assigns it false
# args = parser.parse_args()
# if args.verbose:
#     # display if verbosity is specified, display nothing if not
#     # fails the if statement b/c specifying no argument turns it into a None
#     print("verbosity turned on")

# # short options
# parser.add_argument('-v','--verbose',help='increase output verbosity',
#                     action='store_true')
# args = parser.parse_args()
# if args.verbose:
#     print('verbosity turned on')

# # combining positional and optional arguments
# parser.add_argument('square',type=int,help='display square of given number')
# parser.add_argument('-v','--verbose', action='store_true', help='incrase output verbosity')
# args = parser.parse_args()
# answer = args.square**2
# if args.verbose:
#     print(f'the square of {args.square} equals {answer}')
# else:
#     print(answer)

# # give the ability to have multiple verbosity values
# parser.add_argument('square',type=int,help='display square of given number')
# parser.add_argument('-v','--verbose', type=int, choices=[0,1,2],
#                     help='incrase output verbosity')
# args = parser.parse_args()
# answer = args.square**2
# if args.verbose == 2:
#     print(f'the square of {args.square} equals {answer}')
# elif args.verbose ==1:
#     print(f"{args.square}^2 == {answer}")
# else:
#     print(answer)

# # different approach with verbosity via Count
# parser.add_argument('square', type=int,
#                     help='display the square of a given number')
# parser.add_argument('-v','--verbosity',action='count',
#                     help='increase output verbosity')
# args = parser.parse_args()
# answer = args.square**2
# if args.verbosity == 2:
#     print(f" the square of {args.square} is {answer}")
# elif args.verbosity==1:
#     print(f"{args.square}^2 == {answer}")
# else:
#     print(answer)

# # fix bug with above code
# parser.add_argument('square', type=int,
#                     help='display the square of a given number')
# parser.add_argument('-v','--verbosity',action='count',
#                     default=0, help='increase output verbosity')
# # bugfix: add default=0 to prevent it from defaulting to None
# args = parser.parse_args()
# answer = args.square**2
# if args.verbosity >= 2:  # bugfix: replace == with >=
#     print(f" the square of {args.square} is {answer}")
# elif args.verbosity == 1:
#     print(f"{args.square}^2 == {answer}")
# else:
#     print(answer)

# # get more advanced!
# parser.add_argument('x', type=int, help='the base')
# parser.add_argument('y', type=int, help='the exponent')
# parser.add_argument('-v', '--verbosity', action='count',
#                     default=0, help='incrase verbosity')
# args = parser.parse_args()
# answer = args.x**args.y
# if args.verbosity >=2:
#     print(f"{args.x} to the power {args.y} equals {answer}")
# elif args.verbosity == 1:
#     print(f"{args.x}^{args.y}=={answer}")
# else:
#     print(answer)

# # use verbosity level to display more txt instead of change the text??
# parser.add_argument('x', type=int, help='the base')
# parser.add_argument('y', type=int, help='the exponent')
# parser.add_argument('-v', '--verbosity', action='count',
#                     default=0, help='incrase verbosity')
# args = parser.parse_args()
# answer = args.x**args.y
# if args.verbosity >= 2:
#     print(f"Running '{__file__}'")
# if args.verbosity >= 1:
#     print(f"{args.x}^{args.y} == ",end="")
# print(answer)

# # conflicting arguments
# group = parser.add_mutually_exclusive_group()
# group.add_argument('-v', '--verbose', action='store_true')
# group.add_argument('-q','--quiet',action='store_true')
# parser.add_argument('x', type=int, help='the base')
# parser.add_argument('y', type=int, help='the exponent')
# args = parser.parse_args()
# answer = args.x**args.y
# if args.quiet:
#     print(answer)
# elif args.verbose:
#     print(f"{args.x} to the power {args.y} equals {answer}")
# else:
#     print(f"{args.x}^{args.y}=={answer}")
#
# # conflicting arguments
# parser = ap.ArgumentParser(description='calculate X to the power of Y')
# group = parser.add_mutually_exclusive_group()
# group.add_argument('-v', '--verbose', action='store_true')
# group.add_argument('-q','--quiet',action='store_true')
# parser.add_argument('x', type=int, help='the base')
# parser.add_argument('y', type=int, help='the exponent')
# args = parser.parse_args()
# answer = args.x**args.y
# if args.quiet:
#     print(answer)
# elif args.verbose:
#     print(f"{args.x} to the power {args.y} equals {answer}")
# else:
#     print(f"{args.x}^{args.y}=={answer}")

