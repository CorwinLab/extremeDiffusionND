import numpy as np
# from plotLinePointStats import processLinePointStatsNPY

# TODO: Rewrite as a double sum, then redo analytically
# def g(lambd,alpha=1,n_sims=10000):
#     # (n_sims, 4)
#     # left (-x), down(-y), up(y), right(x)
#     xis = np.random.dirichlet([alpha]*4,size=n_sims)
#     # expectation values....
#     secMoment = np.mean(xis*xis,axis=0)  # E[xi(nhat)^2]
#     negXnegY = np.mean(xis[:,0]*xis[:,1],axis=0)  # E[xi(-x)xi(-y)]
#     negXY = np.mean(xis[:,0]*xis[:,2],axis=0)  # E[xi(-x)(y)]
#     negXX = np.mean(xis[:,0]*xis[:,3],axis=0)  # E[xi(-x)xi(x)]
#     negYY = np.mean(xis[:,1]*xis[:,2],axis=0)  # E[xi(-y)xi(y)]
#     negYX = np.mean(xis[:,1]*xis[:,3],axis=0)  # E[xi(-y)xi(x)]
#     yx = np.mean(xis[:,2]*xis[:,3],axis=0)  # E[xi(x)xi(y)]
#
#     # going to group by 1 of the sums
#     xSums = (secMoment[3]*np.exp(2*lambd) + negXX*np.exp(0)
#              + yx*np.exp(lambd) + negYX*np.exp(lambd))
#     negXSums = (negXX*np.exp(0) + secMoment[0]*np.exp(-2*lambd)
#                  + negXY*np.exp(-lambd) + negXnegY*np.exp(-lambd))
#     ySums = (yx*np.exp(lambd) + negXY*np.exp(-lambd)
#              + secMoment[2]*np.exp(0) + negYY*np.exp(0))
#     negYSums = (negYX*np.exp(lambd) + negXnegY*np.exp(-lambd)
#                 + negYY*np.exp(0) + secMoment[1]*np.exp(0))
#     return np.log(xSums + negXSums + ySums + negYSums) - 2*np.log((np.cosh(lambd)+1)/2)

def g2(l, alpha=1,n_sims=10000):
    xis = np.random.dirichlet(np.array([alpha]*4),size=n_sims).astype(np.quad)
    nhats = np.array([(1,0),(-1,0),(0,1),(0,-1)])  #x, -x, y, -y
    doubleSum = 0
    for idx1,n1 in enumerate(nhats):
        for idx2,n2 in enumerate(nhats):
            print(f"n1, n2: {n1},{n2}")
            doubleSum += np.mean(xis[:,idx1]*xis[:,idx2],axis=0)*np.exp(l*(np.dot((1,0),(n1+n2))))
    return np.log(doubleSum) - 2*np.log((np.cosh(l)+1)/2)

def gAnalytic(l, alpha=1):
    a0 = 4*alpha  # sum of all alphas
    return (np.log(((alpha*(alpha+1))/(a0*(a0+1)))*(2*np.cosh(2*l) + 2)
                  + (alpha**2/(a0*(a0+1)))*(4+4*(2*np.cosh(l))))
            - 2*np.log((np.cosh(l)+1)/2))

if __name__ == "__main__":
    dataDir = "/mnt/talapasData/data/pastLineTake3/1000/Line/"
    # dataDir = "/mnt/talapasData/data/pastLineAlpha01/1000/Line/"
