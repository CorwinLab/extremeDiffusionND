import numpy as np
import npquad
from time import time as wallTime  # start = wallTime() to avoid issues with using time as variable


# the following only work on LOCUST
def productOfDirichletNumbers(n):
    """" calculate the product of n dirichlet numbers"""
    params = np.array([0.1]*4)
    # # # rand_vals = np.random.dirichlet(params,size=n)
    # # # rand_vals = rand_vals.astype(np.quad)
    # # # prod = np.prod(rand_vals[:,0])
    # func = getRandomDistribution("DirichletLocust",params)
    # rand_vals = np.array([func() for i in range(n)])
    # # prod = np.prod(rand_vals[:,0])
    # # return prod
    # prob = np.exp(np.sum(np.log(rand_vals)))
    rand_vals = np.random.dirichlet(params,size=n).astype(np.quad)
    prod = np.exp(np.sum(np.log(rand_vals[:,0])))
    return prod

def getLogP(t):
    """ """
    sumP = (productOfDirichletNumbers(t) + productOfDirichletNumbers(t)
           + productOfDirichletNumbers(t) + productOfDirichletNumbers(t))
    logP = np.log(sumP)
    return logP.astype(float)


def diamondCornerVariance(t):
    """ calculate var[lnP] of rwre being at the 4 corners
    process (equiv of setting v=1 for vt regime):
    take product of t dirichlet numbers, 4 times independently
    repeat a ton of times
    then take log of all of them and calculate variance
    note because log(product) is sum(log) we can add instead?
    for now only going to do this for dirichlet as a check.
    which means lambda_ext will be for dirichlet with alpha=0.1

    That is. calculates variance of diamond corners, all for one time.
    """
    num_samples = 10000
    logPs = []
    startTime = wallTime()
    print(f"starting var calc:")
    for _ in range(num_samples):
        logPs.append(getLogP(t))
    print(f"end of iterations at {wallTime() - startTime}")
    var = np.var(logPs)
    print(f"end of var calc")
    return var

def diamondVarFinal(ts):
    # radii = ts
    params = np.array([0.1]*4)
    # lambda_ext = getExpVarXDotProduct("DirichletLocust",params)
    varLnPs = []
    for t in ts:
        print(f"t: {t}")
        varLnPs.append(diamondCornerVariance(t))
    return np.array(varLnPs)