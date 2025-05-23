import numpy as np
# import npquad
# from time import time as wallTime  # start = wallTime() to avoid issues with using time as variable

def computeVarLnP(t, alpha=1, nSamples=500):
    """ 
    return the variance of ln(P_corners)
    where P_Corners is the sum of 4 indpt products of diirchlet numbers
    
    To avoid npquad, we are going to do this by scaling probability by
    a constant, the mean of the ln(P_individaulcorner)s
    """
    # Generate some data
    data = np.array([ [np.sum(np.log(np.random.dirichlet([alpha]*4, size=t)[:,0])) for _ in range(4)] for _ in range(nSamples)])
    # Find the mean of the logs
    x = (np.max(data) + np.min(data))/2
    return np.var(np.log(np.sum(np.exp(data-x), axis=1)))

def getVLPDiamondList(ts):
    return np.array([computeVarLnP(t) for t in ts])
