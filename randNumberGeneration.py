import numpy as np
from numba import njit, vectorize


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

# the next two functions are specifically for locust
@vectorize
def gammaDist(alpha, scale):
    return np.random.gamma(alpha, scale)

@njit
def randomDirichletLocust(alphas):
    gammas = gammaDist(alphas, np.ones(alphas.shape))
    return gammas / np.sum(gammas)

# return to normal functions
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

@njit
def randomCorner():
    """
    Choose 2 values p1, p2 independently from a distribution on (0,1) with mean 1/2. Use uniform for ease
    but in principle any distribution could be used.
    Assign probabilities as follows:
        up: p1/2; right: (1-p1)/2; down: (1-p2)/2; left: p2/2
    Interpreted as 2 random walks where the choices are up/right, and down/left, as opposed to the
        traditional left/right, and up/down.
    Returns
    -------
    rand_vals: np array of 4 probabilities which sum to 1) in the following order:
        left, down, up, right
    """
    p1 = np.random.uniform(0,1)
    p2 = np.random.uniform(0,1)
    rand_vals = [p2/2, (1-p2)/2, p1/2, (1-p1)/2]
    return rand_vals




def getRandomDistribution(distName, params=''):
    """Get the function to run the random distribution we'll use."""
    # Need to convert numpy array to list to be properly
    # Converted to a string
    if isinstance(params, np.ndarray):
        params = list(params)
    if distName=='DirichletLocust':
        params = np.array(params)
    code = f'random{distName}'
    return eval(f'njit(lambda : {code}({params}))')
