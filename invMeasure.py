import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import colors
from numba import njit

@njit
def iterateInvMeasure(mu, sameSiteTransitionProbs, diffSiteTransitionProbs, t):
    ''' 
    Iterate the invariant measure

    Examples
    --------
    width = sameSiteTransitionProbs.shape[0] // 2
    for t in range(100):
        occ = iterateInvMeasure(occ, sameSiteTransitionProbs, diffSiteTransitionProbs, t)
    
    fig, ax = plt.subplots()
    im = ax.imshow((occ / occ[L, L])[L-10:L+11, L-10:L+11], interpolation=None, norm=colors.LogNorm(vmin=10**-1, vmax=1))
    plt.colorbar(im, ax=ax)
    fig.savefig(f"InvariantMeasure{t}.pdf", bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot((occ / occ[L, L])[L-10:L+11, L])
    fig.savefig(f"Slice{t}.png", bbox_inches='tight')

    invMeasure = alpha / (alpha+1)
    print(invMeasure)
    with np.printoptions(precision=4):
        print(occ[L -5: L+6, L-5:L+6] / occ[L, L])
    '''
    assert np.isclose(np.sum(sameSiteTransitionProbs), 1)
    assert np.isclose(np.sum(diffSiteTransitionProbs), 1)

    L = mu.shape[0] // 2
    width = sameSiteTransitionProbs.shape[0] // 2
    newMu = np.zeros(mu.shape)
    
    for i in range(L-t*width-1, L+t*width+1):
        for j in range(L-t*width-1, L+t*width+1):
            if (i == L) and (j == L):
                newMu[i-width: i+width+1, j-width:j+width+1] += mu[i, j] * sameSiteTransitionProbs
                continue
            newMu[i-width: i+width+1, j-width:j+width+1] += mu[i, j] * diffSiteTransitionProbs
    
    return newMu

def getInvMeasure(sameSiteTransitionProbs, diffSiteTransitionProbs, tMax=100, L=1000):
    occ = np.zeros(shape=(2*L+1, 2*L+1))
    occ[L, L] = 1

    for t in range(tMax):
        occ = iterateInvMeasure(occ, sameSiteTransitionProbs, diffSiteTransitionProbs, t)

    return occ

def localTimeSum(sameSiteTransitionProbs, diffSiteTransitionProbs):
    mu = getInvMeasure(sameSiteTransitionProbs, diffSiteTransitionProbs)
    L = mu.shape[0] // 2

    # Define the jump magnitude and direction
    jumps = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

    localTimeSum = 0
    for lx in range(-10, 11):
        for ly in range(-10, 11):
            absL = np.abs(lx) + np.abs(ly)
            if lx==0 and ly==0:
                probDist = sameSiteTransitionProbs
            else: 
                probDist = diffSiteTransitionProbs
            
            suml = 0 
            for i in jumps: 
                for j in jumps: 
                    step = i + j + np.array([lx, ly])
                    stepDist = np.abs(step[0]) + np.abs(step[1])
                    stepProb = probDist[(i+j)[0] + 2, (i+j)[1] + 2] 
                    suml += (stepDist - absL) * stepProb
            print(suml, lx, ly)
            localTimeSum += suml * mu[L + lx, L + ly] # These might be incorrect
            
    return localTimeSum

if __name__ == '__main__':

    alpha = 0.1
    # Probability distribution of two walks at the same site
    # Probability of jumping in different directions
    Eij = alpha / 4 / (4 * alpha + 1)
    # Probability of jumping in the same direction
    Eii = (alpha + 1) / 4 / (4 * alpha + 1)

    sameSiteTransitionProbs = np.array([[0, 0, Eij, 0, 0],
                                [0, 2 * Eij, 0, 2 * Eij, 0],
                                [Eij, 0, 4 * Eii, 0, Eij],
                                [0, 2 * Eij, 0, 2 * Eij, 0],
                                [0, 0, Eij, 0, 0]])
    
    EiEj = 1/16
    diffSiteTransitionProbs = np.array([[0, 0, EiEj, 0, 0],
                                [0, 2 * EiEj, 0, 2 * EiEj, 0],
                                [EiEj, 0, 4 * EiEj, 0, EiEj],
                                [0, 2 * EiEj, 0, 2 * EiEj, 0],
                                [0, 0, EiEj, 0, 0]])
    
    coeff = localTimeSum(sameSiteTransitionProbs, diffSiteTransitionProbs)
    print(coeff)


    
