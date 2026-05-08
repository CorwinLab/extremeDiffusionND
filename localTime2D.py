import numpy as np


def gAnalytic(l, alpha):
    """ returns g(lambda)"""
    a0 = 4 * alpha  # sum of all alphas
    func = (np.log(((alpha * (alpha + 1)) / (a0 * (a0 + 1))) * (2 * np.cosh(2 * l) + 2)
                   + (alpha ** 2 / (a0 * (a0 + 1))) * (4 + 4 * (2 * np.cosh(l))))
            - 2 * np.log((np.cosh(l) + 1) / 2))
    return func


def localTimeSSRW(v=0.08, alpha=1, tMax=1000, d=2):
    # order: xx, x-x, xy, x-y, -xx, -x-x, -xy, -x-y, yx,y-x,yy,y-y,-yx,-y-x,-yy,-y-y

    # (2d*2d, 2 (# walkers), d)
    # np.array([[[walk1x,walk1y],[walk2x,walk2y]],,,,,,,,,,,,,,,,])
    # order goes walk1 xhat (walk 2s), walk1 -xhat (walk 2s), walk1 yhat (walk2s), walk1 -yhat (walk2s)
    options = np.array([[[+1, 0], [+1, 0]], [[+1, 0], [-1, 0]], [[+1, 0], [0, 1]], [[+1, 0], [0, -1]],
                        [[-1, 0], [1, 0]], [[-1, 0], [-1, 0]], [[-1, 0], [0, 1]], [[-1, 0], [0, -1]],
                        [[0, +1], [1, 0]], [[0, +1], [-1, 0]], [[0, +1], [0, 1]], [[0, +1], [0, -1]],
                        [[0, -1], [1, 0]], [[0, -1], [-1, 0]], [[0, -1], [0, 1]], [[0, -1], [0, -1]]])
    localTime = np.zeros(tMax)
    localTime[0] = 1  # t=0 LT = 0
    # # tilted one point probabilitiy measure
    # initialize at t=0 with walks at 0,0, eqn 16 evaluated at t=0 and r1[0] = r2[0] = vec(0)
    paths = np.zeros((tMax, d, 2))  #time by dimension by # walks
    eqn16 = np.zeros((tMax))
    phi0 = (np.linalg.norm([paths[0, 0, 0] - v * 0 / np.sqrt(1 - 2 * v ** 2), paths[0, 1, 0]])
            * np.linalg.norm([paths[0, 0, 1] - v * 0 / np.sqrt(1 - 2 * v ** 2), paths[0, 1, 1]]))
    eqn16[0] = np.exp(gAnalytic(np.arctanh(2 * v), alpha) * localTime[0]) * phi0

    for t in range(1, tMax):
        # first update the paths
        # if at same site, same jump distribution. thus same tilted distribution
        probs = [1 / 16] * 16

        outcome = np.random.choice(np.arange(16),p=probs)
        vector = options[outcome]
        paths[t, :, :] = paths[t - 1, :, :] + vector  # paths is t by d by #walks

        # local time is how many incidents where the walks are at the same site
        # including this current timestep
        if (paths[t, :, 0] == paths[t, :, 1]).all():
            localTime[t] = localTime[t - 1] + 1
        else:
            localTime[t] = localTime[t - 1]

        # update eqn 16
        phis = (np.linalg.norm([paths[:, 0, 0] - v * t / np.sqrt(1 - 2 * v ** 2), paths[:, 1, 0]])
                * np.linalg.norm([paths[:, 0, 1] - v * t / np.sqrt(1 - 2 * v ** 2), paths[:, 1, 1]]))
        eqn16[t] = np.exp(gAnalytic(np.arctanh(2 * v), alpha) * localTime[t]) * phis
    return paths, eqn16, localTime


def manyIterationsSSRW(n, v=0.08, alpha=1, tMax=1000, d=2):
    """ return the avg. of eqn 16 wrt jacob's shitty tilted measure"""
    localTimes = []
    for i in range(n):
        paths, eqn16, localTime = localTimeSSRW(v=v, alpha=alpha, tMax=tMax, d=d)
        localTimes.append(localTime)
    return localTimes


def correlated2PointMotion(v, alpha):
    """ return the probabilities associated with eqn. 14 of jacob's 2d random walk overleaf doc."""
    # order: xx, x-x, xy, x-y, -xx, -x-x, -xy, -x-y, yx,y-x,yy,y-y,-yx,-y-x,-yy,-y-y

    nhats = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
    a0 = 4 * alpha  # technically it's the sum of the alphas, but all ours are equal
    # the denominator is the sum of the 16 terms
    sameCov = alpha * (alpha + 1) / (a0 * (a0 + 1))
    diffCov = alpha ** 2 / (a0 * (a0 + 1))

    terms = np.array([sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[0]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[1]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[2]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[3]))),

                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[0]))),
                      sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[1]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[2]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[3]))),

                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[0]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[1]))),
                      sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[2]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[3]))),

                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[0]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[1]))),
                      diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[2]))),
                      sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[3])))
                      ])
    normalize = np.sum(terms)
    return terms / normalize


def get2PointVectors(probs):
    """ using probs from correlated2PointMotion, return an option of moves for 2 walks to move"""
    # (2d*2d, 2 (# walkers), d)
    # np.array([[[walk1x,walk1y],[walk2x,walk2y]],,,,,,,,,,,,,,,,])
    # order goes walk1 xhat (walk 2s), walk1 -xhat (walk 2s), walk1 yhat (walk2s), walk1 -yhat (walk2s)
    options = np.array([[[+1, 0], [+1, 0]], [[+1, 0], [-1, 0]], [[+1, 0], [0, 1]], [[+1, 0], [0, -1]],
                        [[-1, 0], [1, 0]], [[-1, 0], [-1, 0]], [[-1, 0], [0, 1]], [[-1, 0], [0, -1]],
                        [[0, +1], [1, 0]], [[0, +1], [-1, 0]], [[0, +1], [0, 1]], [[0, +1], [0, -1]],
                        [[0, -1], [1, 0]], [[0, -1], [-1, 0]], [[0, -1], [0, 1]], [[0, -1], [0, -1]]])
    # need the squeeze to get the output to be (2,2)
    move = options[np.random.choice(np.arange(16), p=probs)]
    return move


def phi(walk_x,walk_y, v, t):
    return np.linalg.norm([(walk_x - v*t)/np.sqrt(1-2*v**2), walk_y])


def version2(v, alpha, tMax, d=2):
    """
    build up 2 random walks in the tilted probability measure acc. to eqn 14
    also calculate the local time of the 2 walks along the way
    """
    localTime = np.zeros(tMax)
    localTime[0] = 1  # fencpost problem
    # # tilted one point probabilitiy measure
    # initialize at t=0 with walks at 0,0, eqn 16 evaluated at t=0 and r1[0] = r2[0] = vec(0)
    walks = np.zeros((tMax, d, 2))  # time by dimension by # walks
    probs = correlated2PointMotion(v, alpha)
    for t in range(1, tMax):
        initialMove = get2PointVectors(probs)
        moveWalk1 = initialMove[0]
        if (walks[t - 1, :, 0] == walks[t - 1, :, 1]).all():
            moveWalk2 = initialMove[1]
        else:
            newMove = get2PointVectors(probs)
            moveWalk2 = newMove[1]
        walks[t, :, 0] = walks[t - 1, :, 0] + moveWalk1
        walks[t, :, 1] = walks[t - 1, :, 1] + moveWalk2

        # update local time
        if (walks[t, :, 0] == walks[t, :, 1]).all():
            # print('they moved to the same site!')
            localTime[t] = localTime[t - 1] + 1
        else:
            localTime[t] = localTime[t-1]
        # print('local time: ',localTime)
    return walks, localTime


def manyVersion2(n, tMax, v=0.08, alpha=1, d=2):
    """ return the avg. of eqn 16 wrt jacob's shitty tilted measure"""
    localTimes = []
    for i in range(n):
        paths, localTime = version2(v=v, alpha=alpha, tMax=tMax, d=d)
        localTimes.append(localTime)
    return localTimes