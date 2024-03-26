import numpy as np
import evolve2DLattice as ev

#can probably put this in a different file
def checkIfMeanTCircular(meanTArrival,band):
    """
    Takes an array of meanTArrival, chooses a band of TArrival and returns  the coordinates of the meanTArrivals
    in the band as polar coords. If a plot of r vs theta is radially symmetric then should get a flat line
    Parameters
        meanTArrival: should be like np.mean(tArrival,0) where tArrival
            is like (#runs,2L+1,2L+1) array. shape of meanTArrival
            should be (2L+1,2L+1)
        band: [lower,upper] of meanTArrival
    returns:
        distance (r) and angle (theta) of the tArrivals within specified band
    """
    cond = ((band[0]<meanTArrival) & (meanTArrival<band[1]))
    L = int(((meanTArrival.shape[0])-1)/2)    # this is the stupidest way of extracting L
    i,j = np.where(cond)
    i, j = i-L,j-L
    r, theta = cartToPolar(i,j)
    return r, theta

#can put in a different file
def plotVarTvsDistance(varT,powerlaw=0):
    """
    Takes in an array of variances of tArrival, calculates their polar coordinates
    Goal: use output to plot varT vs r or theta on a loglog scale
    Parameters:
        varT: array (2L+1,2L+1) in size. should come from like np.nanvar(tArrival,0)
    """
    L = int(((varT.shape[0])-1)/2)
    i, j = np.meshgrid(range(varT.shape[0]),range(varT.shape[0]))
    i, j = i - L, j - L
    r, theta = ev.cartToPolar(i,j)
    return varT, r, theta

#can put in a different file
def tArrivalPastPlane(tArrival,line,axis):
    """
    Define a plane, and ask for the first tArrival past that plane (it should be the smallest tArrival *on* the line)
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
