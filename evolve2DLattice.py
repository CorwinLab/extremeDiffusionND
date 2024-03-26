import numpy as np
# import npquad
import os
from scipy.ndimage import morphology as m
import time
# from numba import jit, njit

# helper functions
def doubleArray(array,arraytype, fillValue = 0):
    """
    Takes an existing numpy array and doubles it, keeping existing vals at center
    Parameters:
        array: the array you want to double, assumes square and odd size
        arraytype: tell the dtype of the array (int, float, etc)
        fillValue: what do you want to fill the new entries with, automatically set to 0
    """
    length = (array.shape[0])//2
    if length == 0:
        newLength = 1
    else:
        newLength = 2 * length
    newArray = np.full((2 * newLength + 1, 2 * newLength + 1), fillValue, dtype=arraytype)
    newArray[newLength-length:newLength+length+1, newLength-length:newLength+length+1] = array
    return newArray
def executeMoves(occupancy, i, j, rng, dirichlet, PDF):
    """
    Moves agents in a 2Dlattice according to dirichlet & multinomial
    Parameters:
        occupancy: the array you are working in, with agents
        i, j = sites with agents in them/sites that are occupied
        rng: numpy random number generator
        dirichlet: boolean; if true uses dirichlet biases; if false uses SSRW
        PDF: boolean; if true then multiplies biases; if false then draws multinomial
    """
    # Generate biases for each site
    if dirichlet:
        #print("Dirichlet")
        biases = rng.dirichlet([1]*4, i.shape[0])
    else:
        #print("SSRW!")
        biases = np.full([i.shape[0],4],1/4)
    # On newer numpy we can vectorize to compute the moves
    if PDF: #if doing PDF then multiply by biases
        moves = occupancy[i,j].reshape(-1,1) * biases #reshape -1 takes the shape of occupancy
    else:
        moves = rng.multinomial(occupancy[i,j], biases)

    # Note that we can use the same array because we're doing checkerboard moves
    # If we want to use a more general jump kernel we need to use a new (empty) copy of the space
    occupancy[i,j-1] += moves[:,0] # left
    occupancy[i+1,j] += moves[:,1] # down
    occupancy[i,j+1] += moves[:,2] # right
    occupancy[i-1,j] += moves[:,3] # up
    occupancy[i,j] = 0 # Remove everything from the original site, as it's moved to new sites
    return occupancy
def changeArraySize(array,size,fillval):
    length = (array.shape[0]) // 2
    newsize = size//2
    if length < size:
        newArray = np.full((2 * newsize + 1, 2 * newsize + 1), fillval, dtype=int)
        newArray[newsize - length:newsize + length + 1, newsize - length:newsize + length + 1] = array
    else:
        newArray = array
    return newArray

# main functions & generators + wrappers
# change to evolve2DLattice
def numpyEvolve2DLatticeAgent(occupancy, maxT,dirichlet, PDF, occtype,startT = 1, rng = np.random.default_rng()):
    """
    a generator object
    evolves agents in a 2Dlattice out to some time MaxT, according to dirichlet &
    multinomial, with dynamic scaling.
    Parameters:
        occupancy: either number (NParticles) or occupancy array
        maxT: timestep you want to go out to
        dirichlet: boolean; if true uses dirichlet biases; if false uses SSRW
        PDF: boolean; if true then multiplies biases; if false then draws multinomial
        occtype: pass through the dtype of the occupancy array (int for agents, float for pdf)
    """
    # Convert a scalar occupancy into a 2d array of size (1,1); commented out because redundant
    # if np.isscalar(occupancy):
    #     occupancy = np.array([[occupancy]], dtype=float)
    for t in range(startT, maxT):
        if t == 1 or t == 100 or t == 1000:
            print("Timestep:", t)
            s = time.time()
        # Find the occupied sites
        i,j = np.where(occupancy != 0)
        # If the occupied sites are at the limits (i.e if min(i,j) = 0 or max(i,j) = size)
        # then we need to enlarge occupancy and create a new array
        # print(f'{occupancy.shape}, {np.min([i,j])}, {np.max([i,j])}')
        if (np.min([i,j]) <= 0) or (np.max([i,j]) >= np.min(occupancy.shape) -1 ):
            occupancy = doubleArray(occupancy,occtype)
            # These next two lines are a waste and we could just do index translation
            sites = (occupancy != 0)
            i,j = np.where(sites)
        occupancy = executeMoves(occupancy, i, j, rng, dirichlet, PDF)
        if t == 1 or t == 100 or t==1000:
            print("Time it took: ",time.time()-s)
        yield t, occupancy

#generateFirstArrivalTime
def generateFirstArrivalTimeAgent(occupancy, maxT,dirichlet,PDF = False, startT =1,):
    """
    Evolves agents in 2DLattice with Dirichlet biases and multinomial sampling
    Returns array of time of first arrivals of each site
    includes dynamic scaling
    Parameters:
        occupancy: initial occupancy, can be a number (NParticles) or an existing array
        maxT: timestep you want to go out to
        dirichlet: Boolean; if True then uses Dirichlet biases; if false uses SSRW
        PDF: boolean; if False (default), does agent-based (multinomial); if true; replaces multinomial w/ multiplication of biases
        startT: optional arg, starts at 1
    """
    notYetArrived = -1
    print("Dirichlet? (genFirstArrival)", dirichlet) #debugging print
    print("PDF (genFirstArrival): ",PDF) #debugging print
    if PDF and occupancy != 1:
        print("Warning: You are trying to evolve a PDF with N!= 1, so it won't be normalized.")
    if PDF:
        occtype = float
    else:
        occtype = int
    if np.isscalar(occupancy):
        occupancy = np.array([[occupancy]],dtype = occtype)
    tArrival = np.copy(occupancy)
    tArrival[:] = notYetArrived
    for t, occ in numpyEvolve2DLatticeAgent(occupancy, maxT,dirichlet, PDF,occtype):
        if tArrival.shape[0] != occ.shape[0]:
            # Note: this is fragile, we assume that doubling tArrival will always work
            tArrival = doubleArray(tArrival,arraytype=int,fillValue = notYetArrived)
        tArrival[(occ > 0) & (tArrival == notYetArrived)] = t
    return tArrival, occ

#can maybe delete this now
#wrapper function for generateFirstArrivalTimeAgent, saves to a path; OLD? ISH?
def runFirstArrivals(occupancy,MaxT,dirichlet,PDF, iterations,directoryName):
    """
    runs iterations of first arrival arrays, saves to a folder thats specified or created
    :param occupancy: existing occupancy array or a number(ie num. of particles)
    :param MaxT: integer; number of timestseps you want to go out to
    :param dirichlet: boolean; if true uses dirichlet biases; if false uses SSRW
    :param PDF: boolean; if true then multiplies biases; if false then draws multinomial
    :param iterations: integer, number of iterations you want to run
    :param directoryName: string, the path  name you want to create to save tArrivals to
    """
    path = f"{directoryName}"
    statsPath = f"{directoryName}"+"Statistics"
    if PDF: #just to better label directories
        path = path + "PDF"
        statsPath = statsPath + "PDF"
    # create a folder to throw all runs into, or check that it exists & move to that folder
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(statsPath)
        #os.chdir(path)
        print(f"{path} and{statsPath} have been created.")
    else:
        #os.chdir(f"{directoryName}")
        print("folder exists")

    #run your iterations and save each tArrival and occ array into the folder created/specified
    for i in range(iterations):
        print("Dirichlet (runFirstArrivals):", dirichlet)
        tArrival, occ = generateFirstArrivalTimeAgent(occupancy,MaxT,dirichlet, PDF)
        np.savez_compressed(f"{path}/{i}.npz",tArrival=tArrival,occ=occ)

        if not PDF: #if Agent (ie not PDF) then save stats; otherwise don't save them (for testing)
            perimeter, area, time, roughness, avgDist, avgDist2 = getRoughness(tArrival)
            # and save the analysis quantities to statspath/systid.npz
            np.savez_compressed(f"{statsPath}/{i}.npz", perimeter=perimeter, area=area, time=time, roughness=roughness,
                            avgBoundaryDist=avgDist, avgBoundaryDist2=avgDist2)
        print(i)
    #move up one directory (back to Documents/code/extremeDiffusionND

    # maybe I should add a running text file that lists the runs I've done
    # and their directory names, their  initial occupancy, MaxT, and iterations?
    #os.chdir("..")


# wrapper for numpyEvolve2DLattice
def run2dAgent(occupancy, maxT):
    for t, occ in numpyEvolve2DLatticeAgent(occupancy, maxT):
        pass
    return t, occ

#data analysis functions
# take a path with files from runFirstArrivals and get mean, var, mask of tArrivals
# this calculates mean and var by hand when loading in every tArrival will take too much memory
def getTArrivalMeanAndVar(path):
    filelist = sorted(os.listdir(path))
    #print(filelist)
    #print(len(filelist))
    notYetArrived = -1
    #this is so dumb. all files should be named "number.npz" so
    #print(filelist)
    # initialize the moments & mask
    # Can I do:
    #tArrMom1, tArrMom2, goodData = None, None, None
    tArrMom1 = None #moment 1 is just t
    tArrMom2 = None #moment 2 is t**2
    goodData = None #the mask
    # go through each file and pull out the tArrival array
    for file in filelist:
        # this should return ONE tArrival array
        tArrival = np.load(f'{path}/{file}')['tArrival']
        # if you are on the first file, make the moment arrays using the first file
        if tArrMom1 is None:
            tArrMom1 = tArrival
            tArrMom2 = tArrival**2
            goodData = (tArrival != -1)
        # if you are somewhere in the middle of the list, first check that the array sizes will be the same
        # or alternatively make them the same
        else:
            if tArrMom1.shape[0] < tArrival.shape[0]:
                # if not the same size, change the moment arrays to be the same size
                # as the incoming tArrival array
                tArrMom1 = changeArraySize(tArrMom1,tArrival.shape[0],notYetArrived)
                tArrMom2 = changeArraySize(tArrMom2,tArrival.shape[0],notYetArrived)
                goodData = changeArraySize(goodData,tArrival.shape[0],notYetArrived)
            if tArrMom1.shape[0] > tArrival.shape[0]:
                # if not the same size, change the moment arrays to be the same size
                # as the incoming tArrival array
                tArrival = changeArraySize(tArrival,tArrMom1.shape[0],notYetArrived)
            #now cumulatively add the moments
            #need to deal with the -1 thing... my variance has a range of -20 to 0
            goodData *= (tArrival != notYetArrived)
            tArrMom1 += tArrival*goodData
            tArrMom2 += (tArrival*goodData)**2
    finalMom1 = tArrMom1/len(filelist)
    finalMom2 = tArrMom2/len(filelist)
    # Return the mean and the variance
    return finalMom1, finalMom2 - finalMom1**2,goodData

#goes inside checkIfMeanTCircular, and plotVarTvsDistance
def cartToPolar(i,j):
    """
    Can take indices (i,j) and turn them into polar coords. r, theta
    Note: indices need to be already shifted so origin is at center appropriately
    """
    r = np.sqrt(i**2+j**2)
    theta = np.arctan2(j,i)
    return r, theta

#can probably put this in a different file
def checkIfMeanTCircular(meanTArrival,band):
    """
    Takes an array of meanTArrival, chooses a band of TArrival
    and plots the coordinates of the meanTArrivals in the band as
    polar coords, i.e plots theta, r.
    If radially symmetric (circularly?) then should get a flat line
    Parameters
    meanTArrival: should be like np.mean(tArrival,0) where tArrival
        is like (#runs,2L+1,2L+1) array. shape of meanTArrival
        should be (2L+1,2L+1)
    band: [lower,upper] of meanTArrival
    returns:
        distance (r) and angle (theta) of
    """
    cond = ((band[0]<meanTArrival) & (meanTArrival<band[1]))
    L = int(((meanTArrival.shape[0])-1)/2)    # this is the stupidest way of extracting L
    i,j = np.where(cond)
    # anyway it shifts coords so oriign @ center
    i, j = i-L,j-L
    r, theta = cartToPolar(i,j)
    return r, theta

#can put in a different file
def plotVarTvsDistance(varT,powerlaw=0):
    """
    Plots the variance of tArrival as a function of distance from origin
    on a loglog scale
    Parameters:
        varT: array (2L+1,2L+1) in size. should come from like np.nanvar(tArrival,0)
        powerlaw: automatically set to 0, if not 0 then can also plot a guessed powerlaw
    """
    L = int(((varT.shape[0])-1)/2)
    i, j = np.meshgrid(range(varT.shape[0]),range(varT.shape[0]))
    i, j = i - L, j - L
    r, theta = cartToPolar(i,j)
    # plt.ion()
    # fig, ax = plt.subplots()
    # ax.set_xlabel("Distance from origin")
    # ax.set_ylabel("Var(TArrival)")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # ax.plot(r.flatten(),varT.flatten(),'.')
    # if powerlaw != 0:
    #     x=np.logspace(1,2)
    #     ax.plot(x,1e-3*x**powerlaw)
    # plt.show()

#can put in a different file
def tArrivalPastPlane(tArrival,line,axis):
    """
    Define a plane, and ask for the first tArrival past that plane
    it should be the smallest tArrival *on* the line*
    Parameters:
        tArrival: an individual run of generateFirstArrivalTime, and
            be a (2L+1,2L+1) shape
        line: idk i need to figure out how to define a line
        axis: i or j; if i then the line drawn is i=line, if j then line is j=line

    Example to get first crossing past plane as a function of plane location, where
    tArrival[1] is one instance of generateFirstArrivalTime
    iset = [i-length for i in range(2*length+1)]
    plt.loglog(iset,[ev.tArrivalPastPlane(tArrival[1],i,'i') for i in range(401)],'.')
    plt.plot(x,0.2*x**1.8)
    """
    if axis == 'i':
        # choose all sites with i=line
        sites = tArrival[line,:]
    elif axis == 'j':
        sites = tArrival[:,line]
    #find the minimum tArrival value in that set of sites
    firstCrossing = np.nanmin(sites)
    return firstCrossing

#roughness statistics functions
#to go inside getRoughness; finds roughness params for a single array at specific tau
def getPerimeterAreaTau(tArrivalArray,tau):
    """
    Calculate the surface roughness of tArrivals as a function of time
    by (number of pixels on boarder)/(total pixels reached)^1/2
    If perfectly smooth, min roughness is like 2*(pi)^1/2
    Ask how roughness scales with tau? (ie time at which you ask for border)
    :param tArrivalArray: array of tArrivals
    :return: a number (roughness)
    Can be put inside something like:
        plt.plot([i for i in range(0,np.max(newTArrival),50)],[ev.measureRoughness(newTArrival,i) for i in range(0,np.max(newTArrival),50)])
    """
    notYetArrived = -1
    mask = ((tArrivalArray<=tau) & (tArrivalArray> notYetArrived))
    boundary = (mask^ m.binary_erosion(mask))
    # get distance to origin of boundary points
    i, j = np.where(boundary)
    L = tArrivalArray.shape[0]//2
    i, j = i - L, j- L
    r, theta = cartToPolar(i,j)
    boundaryR = np.mean(r)
    boundaryR2 = np.mean(np.square(r))
    perimeter = np.sum(boundary)
    area = np.sum(mask)
    return perimeter,area,tau,boundaryR,boundaryR2

#find roughness stuff (perimeter, area, etc) for a single tArrival array
#avgdist = <r> avgdist2 = <r^2>
def getRoughness(tArrivalArray):
    """
    Takes one tArrival arary and returns arrays of perimeter, area, time, and roughness
    :param tArrivalArray: numpy array generated from generateFirstArrivalTimeAgent
    :return:
    """
    perimeter = []
    area = []
    time = []
    avgDist = []
    avgDist2 = []
    # for each file, pull out their perimeter, area, and time and append
    # note that I think this assumes that the diffused particles
    # aren't touching the edge
    for i in range(1, np.max(tArrivalArray)):
        # save as np. arary instead?
        p, a, t,d,d2 = getPerimeterAreaTau(tArrivalArray, i)
        perimeter.append(p);
        area.append(a);
        time.append(t)
        avgDist.append(d)
        avgDist2.append(d2)
    roughness = perimeter/np.sqrt(area)
    return  np.array(perimeter), np.array(area), np.array(time), np.array(roughness), np.array(avgDist),np.array(avgDist2)









#find roughness stats for a directory of tArrivals
def getRoughnessMeanVar(path):
    """
    Take directory of tArrivals, returns list of perimeters, areas, and also
    returns roughnessMean and roughnessVar?
    :param path: directory name of tArrivals
    :return: np array of perimeters, areas, and time, mean roughness, var roughness
    """
    filelist = sorted(os.listdir(path))
    #initialize array of Roughness
    PerimeterList = []
    AreaList = []
    TimeList = []
    RoughnessList = []
    for file in filelist:
        tArrival = np.load(f'{path}/{file}')['tArrival']
        tempP,tempA,tempT, tempR = getRoughness(tArrival)
        #for each file append the arrays of p,a,t to the list
        PerimeterList.append(tempP)
        AreaList.append(tempA)
        TimeList.append(tempT)
        RoughnessList.append(tempR)
    #pre-emptively just set t to be what it should, calc mean and var of roughness
    TimeList = np.mean(TimeList,0)
    rMean = np.mean(RoughnessList,0)
    rVar = np.var(RoughnessList,0)
    #calculate roughness
    return np.array(PerimeterList),np.array(AreaList),np.array(TimeList), rMean,rVar




#can delete since i've implemented PDF in the new function?
#Not quite old but its the slow PDF evolution
# evolves the PDF without dynamic scaling or multinomial
def evolve2DLatticePDF(Length, NParticles, MaxT=None):
    """
    Create a (2Length+1) square lattice with N particles at the center, and let particles diffuse according to
    dirichlet biases in cardinal directions. This evolves the PDF.
    WITHOUT DYNAMIC SCALING OR MULTINOMIAL
    Parameters:
        Length: distance from origin to side of lattice
        NParticles: number of total particles in system
        MaxT: automatically set to be the maximum time particles can evolve to, but can set a specific time
    """
    # automatically tells it the time you can evolve to
    if not MaxT:
        MaxT = Length+1
    # initialize the array, particles @ origin, and the checkerboard pattern
    occupancy = np.zeros((2*Length+1, 2*Length+1))
    origin = (Length, Length)
    occupancy[origin] = NParticles
    i,j = np.indices(occupancy.shape)
    checkerboard = (i+j+1) % 2
    # evolve in time
    for t in range(1,MaxT):
        # Compute biases for every cell within area we're evolving to all at once
        #[[[left,down,right,up]]]
        biases = np.random.dirichlet([1]*4, (2*t-1, 2*t-1))
        # Define interior lattice size to evolve based on timestep
        startPoint = Length-t+1
        endPoint = Length+t
        # save old occupancy, for calculation reasons
        oldOccupancy = occupancy[startPoint:endPoint, startPoint:endPoint].copy()
        # fill occupancy + zero out the old ones
        occupancy[startPoint:endPoint, startPoint-1:endPoint-1] += oldOccupancy * biases[:,:,0]
        occupancy[startPoint+1:endPoint+1, startPoint:endPoint] += oldOccupancy * biases[:,:,1]
        occupancy[startPoint:endPoint, startPoint+1:endPoint+1] += oldOccupancy * biases[:,:,2]
        occupancy[startPoint-1:endPoint-1, startPoint:endPoint] += oldOccupancy * biases[:,:,3]
        occupancy[checkerboard== (t % 2)] = 0
        yield t, occupancy
        # # I'm leaving this code here because it does a better job of explaining what our goal is
        # for i in range(startPoint, endPoint):
        #     #across
        #     for j in range(startPoint, endPoint):
        #         # Do the calculation if the site and the time have opposite parity
        #         if (i + j + t) % 2 == 1:
        #             localBiases = biases[i-startPoint, j-endPoint, :]
        #             # left
        #             occupancy[i, j - 1] += occupancy[i, j] * localBiases[0]
        #             # down
        #             occupancy[i + 1, j] += occupancy[i, j] * localBiases[1]
        #             # right
        #             occupancy[i, j + 1] += occupancy[i, j] * localBiases[2]
        #             # up
        #             occupancy[i - 1, j] += occupancy[i, j] * localBiases[3]
        #             # zero the old one
        #             occupancy[i, j] = 0
    # return occupancy

