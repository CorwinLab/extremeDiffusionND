import numpy as np
import npquad

def evolve(occ, rng):
    i, j = np.where(occ != 0)
    biases = rng.dirichlet([1] * 4, i.shape[0]).astype(np.quad)
    # print(biases.shape)
    # biases = np.zeros((i.shape[0], 4), dtype=np.quad) + np.quad('.25')
    # print(biases.shape)
    print(occ[i,j].reshape(-1,1).shape, biases.shape)
    for index in range(biases.shape[0]):
        occ[i[index], j[index] - 1] += occ[i[index], j[index]] * biases[index, 0]  # left
        occ[i[index] + 1, j[index]] += occ[i[index], j[index]] * biases[index, 1]  # down
        occ[i[index], j[index] + 1] += occ[i[index], j[index]] * biases[index, 2]  # right
        occ[i[index] - 1, j[index]] += occ[i[index], j[index]] * biases[index, 3]  # up
    # occ[i, j] = 0  # Remove everything from the original site, as it's moved to new sites
    return occ

def simpleSSRWCDF(maxTime):
    occ = np.zeros((maxTime*2 + 1, maxTime*2+1), dtype=np.quad)
    occ[maxTime, maxTime] = 1
    cdf = occ.copy()
    rng = np.random.default_rng()
    for i in range(1,maxTime):
        occ = evolve(occ, rng)
        cdf += occ
    return occ, cdf


def productOfDirichletNumbers(n):
    """" this works"""
    params = np.array([0.1]*4)
    rand_vals = np.random.dirichlet(params,size=n)
    rand_vals = rand_vals.astype(np.quad)
    print('before product')
    # prod = np.prod(rand_vals[:,0])
    # prod = np.exp(np.sum(np.log(rand_vals[:,0])))

    # testing to see if np prod is the issue
    prod = np.quad('5')
    print('after product')
    return prod

def getLogP(t):
    """ this works"""
    print('before sum')
    sumP = (productOfDirichletNumbers(t) + productOfDirichletNumbers(t)
           + productOfDirichletNumbers(t) + productOfDirichletNumbers(t))
    print('after sum, before log')
    logP = np.log(sumP)
    print('after log')
    return logP.astype(float)

def manySamples(t,nsamples):
    """ this core dumps"""
    logPs = []
    for _ in range(nsamples):
        logPs.append(getLogP(t))
    return np.array(logPs)

def diamondVar(t):
    """ calculate var[lnP] of rwre being at the 4 corners
    process (equiv of setting v=1 for vt regime):
    take product of t dirichlet numbers, 4 times independently
    repeat a ton of times
    then take log of all of them and calculate variance
    note because log(product) is sum(log) we can add instead?
    for now only going to do this for dirichlet as a check.
    which means lambda_ext will be for dirichlet with alpha=0.1
    """
    logProbs = []
    num_samples = 2
    for _ in range(num_samples):
        # 4 directions
        probSum = (productOfDirichletNumbers(t) + productOfDirichletNumbers(t) +
                   productOfDirichletNumbers(t) + productOfDirichletNumbers(t))
        probs.append(probSum)
        logP = np.log(probSum)
        logProbs.append(logP)
    var = np.var(logProbs)
    return probs, logProbs, var
