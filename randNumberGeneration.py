import numpy as np
from numba import njit


@njit
def randomDelta():
    """
    choose 2 out of 4 directions at random and set those directions to move with
    prob = 1/2 each.
    """
    biases = np.array([0, 0, 0.5, 0.5])
    np.random.shuffle(biases)
    return biases


@njit
def randomOneQuarter():
    biases = np.array([0.25, 0.25, 0.25, 0.25])
    return biases


@njit
def randomDirichlet(alphas):
    return np.random.dirichlet(alphas)


@njit
def randomSymmetricDirichlet(alphas):
    """
    Create a dirichlet distribution which is symmetric about its center
    """
    rand_vals = np.random.dirichlet(alphas)
    return (rand_vals + np.flip(rand_vals)) / 2


@njit
def randomLogNormal(params):
    rand_vals = np.random.lognormal(params[0], params[1], size=4)
    return rand_vals / np.sum(rand_vals)


@njit
def randomLogUniform(params):
    randVals = np.exp(np.random.uniform(-params[0], params[0], size=4))
    return randVals / np.sum(randVals)


def getRandomDistribution(distName, params=''):
    """Get the function to run the random distribution we'll use."""
    # Need to convert numpy array to list to be properly
    # Converted to a string
    if isinstance(params, np.ndarray):
        params = list(params)

    code = f'random{distName}'
    return eval(f'njit(lambda : {code}({params}))')
