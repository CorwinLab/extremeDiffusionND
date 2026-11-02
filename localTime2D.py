import numpy as np
from differenceRandomWalk import twoWalkerTransitionProbabilities, makeDirectionList, computeKappaMuProduct
from scipy.special import logsumexp as LSE
from matplotlib import pyplot as plt
import os
from time import time as wallTime
import json
from numba import njit
import glob


# TODO: DEBUG THIS
def ltSSRWNew(tMax, d=2):
    localTime = np.full(tMax, np.nan)
    localTime[0] = 1
    walks = np.zeros((tMax, d, 2))  # t by (x or y) by walk
    n1 = makeDirectionList(d)
    n2 = makeDirectionList(d)
    d1 = []
    d2 = []
    for n11 in n1:
        for n22 in n2:
            d1.append(n11)
            d2.append(n22)
    d1 = np.array(d1)
    d2 = np.array(d2)
    options = np.stack((d1, d2), axis=1)
    #indices = []
    probs = [1 / 16] * 16  # equal probs
    # print("new implementation probs: ",probs)
    for t in range(1, tMax):
        newMove = get2PointVectors(probs, options)
        # print("new move: ",newMove)
        #indices.append(index)
        # walks[t, :, 0] = walks[t - 1, :, 0] + newMove[0]  # update walk 1
        # walks[t, :, 1] = walks[t - 1, :, 1] + newMove[1]  # update walk 2
        # walks[t-1, 0,:] updates walk 1 for x and y; walks[t-1, 1,:] updates walk 2 for x and y
        # TODO:  no this is wrong!! because i want to add n1hat to walk1 and n2hat to walk2
        # TODO: ok i think i fixed it but now I have to check it against the old implementation again
        walks[t, :, 0] = walks[t - 1, :, 0] + newMove[0,:]  # newmove[0,:] is n1hat  # so we add walk1 + n1hat
        walks[t, :, 1] = walks[t - 1, :, 1] + newMove[1,:]   #newMove[1,:] is n2hat  # do walk2 + n2hat
        # now that the move has happened, check if we need to update local time
        if (walks[t, :, 0] == walks[t, :, 1]).all():  # if they are at the same site, add 1 to the local time
            # print(f"updating local time: localTime[t] = {localTime[t-1]} + 1")
            localTime[t] = localTime[t - 1] + 1
        else:  # if they are at different sites, the local time does not change value
            localTime[t] = localTime[t - 1]
    return np.array(walks), localTime  #, indices


def manySSRW(n, tMax, d=2):
    localTimes = []
    for i in range(n):
        _, localTime = ltSSRWNew(tMax=tMax)
        localTimes.append(localTime)
    meanLT = np.mean(np.array(localTimes), axis=0)
    logMeanExpLT = LSE(np.array(localTimes), axis=0) - np.log(n)
    return meanLT, logMeanExpLT


# def localTimeSSRW(v=0.5, alpha=1, tMax=1000, d=2):
#     # order: xx, x-x, xy, x-y, -xx, -x-x, -xy, -x-y, yx,y-x,yy,y-y,-yx,-y-x,-yy,-y-y
#
#     # (2d*2d, 2 (# walkers), d)
#     # np.array([[[walk1x,walk1y],[walk2x,walk2y]],,,,,,,,,,,,,,,,])
#     # order goes walk1 xhat (walk 2s), walk1 -xhat (walk 2s), walk1 yhat (walk2s), walk1 -yhat (walk2s)
#     options = np.array([[[+1, 0], [+1, 0]], [[+1, 0], [-1, 0]], [[+1, 0], [0, 1]], [[+1, 0], [0, -1]],
#                         [[-1, 0], [1, 0]], [[-1, 0], [-1, 0]], [[-1, 0], [0, 1]], [[-1, 0], [0, -1]],
#                         [[0, +1], [1, 0]], [[0, +1], [-1, 0]], [[0, +1], [0, 1]], [[0, +1], [0, -1]],
#                         [[0, -1], [1, 0]], [[0, -1], [-1, 0]], [[0, -1], [0, 1]], [[0, -1], [0, -1]]])
#     localTime = np.zeros(tMax)
#     localTime[0] = 1  # t=0 LT = 0
#     # # tilted one point probabilitiy measure
#     # initialize at t=0 with walks at 0,0, eqn 16 evaluated at t=0 and r1[0] = r2[0] = vec(0)
#     paths = np.zeros((tMax, d, 2))  #time by dimension by # walks
#     #eqn16 = np.zeros((tMax))
#     #phi0 = (np.linalg.norm([paths[0, 0, 0] - v * 0 / np.sqrt(1 - 2 * v ** 2), paths[0, 1, 0]])
#     #        * np.linalg.norm([paths[0, 0, 1] - v * 0 / np.sqrt(1 - 2 * v ** 2), paths[0, 1, 1]]))
#     #eqn16[0] = np.exp(gAnalytic(np.arctanh(2 * v), alpha) * localTime[0]) * phi0
#
#     for t in range(1, tMax):
#         # first update the paths
#         # if at same site, same jump distribution. thus same tilted distribution
#         probs = [1 / 16] * 16
#
#         outcome = np.random.choice(np.arange(16),p=probs)
#         vector = options[outcome]
#         paths[t, :, :] = paths[t - 1, :, :] + vector  # paths is t by d by #walks
#
#         # local time is how many incidents where the walks are at the same site
#         # including this current timestep
#         if (paths[t, :, 0] == paths[t, :, 1]).all():
#             localTime[t] = localTime[t - 1] + 1
#         else:
#             localTime[t] = localTime[t - 1]
#
#         # update eqn 16
#         #phis = (np.linalg.norm([paths[:, 0, 0] - v * t / np.sqrt(1 - 2 * v ** 2), paths[:, 1, 0]])
#         #        * np.linalg.norm([paths[:, 0, 1] - v * t / np.sqrt(1 - 2 * v ** 2), paths[:, 1, 1]]))
#         #eqn16[t] = np.exp(gAnalytic(np.arctanh(2 * v), alpha) * localTime[t]) * phis
#     return paths, localTime #eqn16, localTime
#
#
# def manyIterationsSSRW(n, v=0.5, alpha=1, tMax=1000, d=2):
#     """ return the avg. of eqn 16 wrt jacob's shitty tilted measure"""
#     localTimes = []
#     for i in range(n):
#         _, localTime = localTimeSSRW(v=v, alpha=alpha, tMax=tMax, d=d)
#         localTimes.append(localTime)
#     return localTimes


# def correlated2PointMotion(v, alpha):
#     """ return the probabilities associated with eqn. 14 of jacob's 2d random walk overleaf doc."""
#     # order: xx, x-x, xy, x-y, -xx, -x-x, -xy, -x-y, yx,y-x,yy,y-y,-yx,-y-x,-yy,-y-y
#
#     nhats = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
#     a0 = 4 * alpha  # technically it's the sum of the alphas, but all ours are equal
#     # the denominator is the sum of the 16 terms
#     sameCov = alpha * (alpha + 1) / (a0 * (a0 + 1))
#     diffCov = alpha ** 2 / (a0 * (a0 + 1))
#
#     terms = np.array([sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[0]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[1]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[2]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[0] + nhats[3]))),
#
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[0]))),
#                       sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[1]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[2]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[1] + nhats[3]))),
#
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[0]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[1]))),
#                       sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[2]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[2] + nhats[3]))),
#
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[0]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[1]))),
#                       diffCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[2]))),
#                       sameCov * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[3])))
#                       ])
#     normalize = np.sum(terms)
#     return terms / normalize


def get2PointVectors(probs, options):
    """ using probs from correlated2PointMotion, return an option of moves for 2 walks to move"""
    # (2d*2d, 2 (# walkers), d)
    # np.array([[[walk1x,walk1y],[walk2x,walk2y]],,,,,,,,,,,,,,,,])
    # order goes walk1 xhat (walk 2s), walk1 -xhat (walk 2s), walk1 yhat (walk2s), walk1 -yhat (walk2s)
    # options = np.array([[[+1, 0], [+1, 0]], [[+1, 0], [-1, 0]], [[+1, 0], [0, 1]], [[+1, 0], [0, -1]],
    #                     [[-1, 0], [1, 0]], [[-1, 0], [-1, 0]], [[-1, 0], [0, 1]], [[-1, 0], [0, -1]],
    #                     [[0, +1], [1, 0]], [[0, +1], [-1, 0]], [[0, +1], [0, 1]], [[0, +1], [0, -1]],
    #                     [[0, -1], [1, 0]], [[0, -1], [-1, 0]], [[0, -1], [0, 1]], [[0, -1], [0, -1]]])
    move = options[np.random.choice(np.arange(16), p=probs)]
    return move


def newTiltedLocalTime(tMax, v, alpha, d=2):
    localTime = np.full(tMax, np.nan)
    localTime[0] = 1  # at t=0, walkers start together so at t=1, local time val  = 1
    walks = np.zeros((tMax, d, 2))  # time by dimension by walk#
    # the tilted measure probabilities aren't dependent on time or location
    d1, d2, correlatedProbs = twoWalkerTransitionProbabilities(alpha, v, correlated=True)
    _, _, uncorrelatedProbs = twoWalkerTransitionProbabilities(alpha, v, correlated=False)
    options = np.stack((d1, d2), axis=1)
    #  print('size of options', np.shape(options))
    for t in range(1, tMax):
        if (walks[t - 1, :, 0] == walks[t - 1, :, 1]).all():
            # print(f"t={t}, at same spot!")
            newMove = get2PointVectors(correlatedProbs, options)
            # print("new move: ", newMove)
        else:
            newMove = get2PointVectors(uncorrelatedProbs, options)
        # recall walks is t by d by #walks
        # walks[t-1, 0,:] updates walk 1 for x and y; walks[t-1, 1,:] updates walk 2 for x and y
        walks[t, :, 0] = walks[t - 1, :, 0] + newMove[0, :]  # update walk 1
        walks[t, :, 1] = walks[t - 1, :, 1] + newMove[1, :]  # update walk 2
        # now that the move has happened, check if we need to update local time
        if (walks[t, :, 0] == walks[t, :, 1]).all():  # if they are at the same site, add 1 to the local time
            # print(f"updating local time: localTime[t] = {localTime[t-1]} + 1")
            localTime[t] = localTime[t - 1] + 1
        else:  # if they are at different sites, the local time does not change value
            localTime[t] = localTime[t - 1]
    return np.array(walks), localTime


def manyNewTiltedLocalTimes(n, tMax, v, alpha, d=2):
    localTimes = []
    for i in range(n):
        _, localTime = newTiltedLocalTime(tMax=tMax, v=v, alpha=alpha, d=d)
        localTimes.append(localTime)
    meanLT = np.mean(np.array(localTimes), axis=0)
    # TODO:  need to include the sum(kappa mu product) in the logmeanExp thing...
    _, KMProd = computeKappaMuProduct(alpha, v, dMax=100)
    logMeanExpLT = LSE(np.array(localTimes), axis=0) - np.log(n)
    logMeanExpLTScaled = LSE(np.array(localTimes) * np.sum(KMProd/np.sqrt(2)), axis=0) - np.log(n)
    return meanLT, logMeanExpLT, logMeanExpLTScaled, KMProd


def runManyPairs(n, tMax, v, alphaList=np.array([10000, 31, 10, 1, 0.1, 0.01, 0.001]),
                 savePath="/home/fransces/Documents/data/localTimes"):
    """ goal of this function is to see how long it takes to run n pairs
    for different alphas"""
    topDir = os.path.join(savePath, f"nSys{n}")
    os.makedirs(topDir, exist_ok=True)
    print(f"savepath: {topDir} \n n={n}")
    for alpha in alphaList:
        print(f"alpha: {alpha}")
        start = wallTime()
        mLT, LMELT, LMELTS, KMProd = manyNewTiltedLocalTimes(n, tMax, v, alpha)
        runTime = wallTime() - start
        print(f"time: {runTime}")
        fileName = os.path.join(topDir, f"localTimesAlpha{alpha}.npz")
        print(f"fileName: {fileName}")
        np.savez_compressed(fileName, data=np.array([mLT, LMELT, LMELTS]), KMProd=KMProd, runTime=runTime,
                            alpha=alpha, v=v, tMax=tMax, n=n)
    return


# def pairwiseManyTiltedLocalTimes(n, tMax, v, alpha, d=2):
#     """ build up local times for every pair of walkers"""
#     walks = np.zeros((tMax, d, n))  # (time by x or y by # walk
#     d1, d2, correlatedProbs = twoWalkerTransitionProbabilities(alpha, v, correlated=True)
#     _, _, uncorrelatedProbs = twoWalkerTransitionProbabilities(alpha, v, correlated=False)
#     options = np.stack((d1, d2), axis=1)
#     localTimes = np.full((tMax, n**2), np.nan)  # instead of just a 1d array, I need n^2 because i'm taking pairs
#     localTimes[0,: ] = 1  # set the t=0 LT to be 1 for every pair
#     # build up the  n walks AND local times
#     for t in range(1, tMax):
#


# def saveAndPlotLocalTimes(nsys, tMax, alphaList=np.array([31, 10, 1, 0.1, 0.01, 0.001]),
#                    v=0.8, d=2, dMax=100,
#                    savePath='/home/fransces/Documents/Figures'):
#     fig, ax = plt.subplots()
#     fig2, ax2 = plt.subplots()
#     ax.set_xlabel('t')
#     ax.set_ylabel('mean tilted local time * sum(kappa*mu)')
#     ax.set_title(f"nsys={nsys}, v={v}")
#     ax2.set_xlabel('t')
#     ax2.set_ylabel("log[Mean[e^(tilted local time * sum(kappa*mu))]]")
#     ax2.set_title(f"nsys={nsys}, v={v}")
#     for alpha in alphaList:
#         # TODO: replace the below function with the new pairwise one
#         mLT, LMELT, LMELTS = manyNewTiltedLocalTimes(nsys, tMax, v, alpha, d=d)
#         _, KMProd = computeKappaMuProduct(alpha, v, dMax=dMax)
#         ax.loglog(mLT*np.sum(KMProd),label=f"alpha={alpha}")
#         ax2.plot(LMELTS,label=f"alpha={alpha}")
#     # TODO: insert SSRW line here
#     fig.legend()
#     fig2.legend()
#     fig.savefig(os.path.join(savePath,"meanLTScaledVsT.png"))
#     fig2.savefig(os.path.join(savePath,"logMeanExpLTScaledVsT.png"))
#     return


def main(topDir="/home/fransces/Documents/data/localTimes/nSys10000/"):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    files = glob.glob(os.path.join(topDir, "*.npz"))
    fig, ax = plt.subplots(figsize=(5,5),dpi=300, constrained_layout=True)
    ax.set_xlabel(r"$t/\left(\frac{1}{\alpha}+1\right)$")
    ax.set_ylabel(r"$\mathbb{E}\left[\mathrm{lt}_{\mathrm{tilt}}\right]\sum{\frac{1}{\sqrt{2}}\kappa\mu}$")

    fig2, ax2, = plt.subplots(figsize=(5,5),dpi=300, constrained_layout=True)
    ax2.set_xlabel(r"$t  $")
    ax2.set_ylabel(r"$\log{\left(\mathbb{E}\left[e^{\mathrm{lt}_{\mathrm{tilt}}}\right]\right)}$")

    fig3, ax3 = plt.subplots(figsize=(5,5),dpi=300, constrained_layout=True)
    ax3.set_xlabel(r"$t / \left(\frac{1}{\alpha}+1\right)$")
    ax3.set_ylabel(r"$\log{\left(\mathbb{E}\left[e^{\mathrm{lt}_{\mathrm{tilt}}\sum{\frac{1}{\sqrt{2}}\kappa\mu}}\right]\right))}$")

    fig4, ax4 = plt.subplots(figsize=(5,5),dpi=300, constrained_layout=True)
    ax4.set_xlabel(r"$t  $")
    ax4.set_ylabel(r"$\mathbb{E}\left[\mathrm{lt}_{\mathrm{tilted}}\right]\sum{\frac{1}{\sqrt{2}}\kappa\mu}$")
    ts = np.arange(1000)
    for file in files:
        temp = np.load(file)
        v = temp['v']
        alpha = temp['alpha']
        if np.isfinite(alpha):  # if we do NOT have the SSRW:
            mLT, LMELT, LMELTS = temp['data'][0, :], temp['data'][1, :], temp['data'][2, :]
            _, KMProd = computeKappaMuProduct(alpha, v, dMax=100)
            tPrefactor = (1/alpha) + 1
            ax.loglog(ts / tPrefactor , mLT * np.sum(KMProd/np.sqrt(2)), label=fr"$\alpha={alpha}$")
            ax2.loglog(ts, LMELT, label=fr"$\alpha={alpha}$")
            ax3.loglog(ts / tPrefactor, LMELTS, label=fr"$\alpha={alpha}$")
            ax4.loglog(ts, mLT * np.sum(KMProd/np.sqrt(2)), label=fr"$\alpha={alpha}$")
        else:  # SSRW
            mLT, LMELT, LMELTS = temp['data'][0, :], temp['data'][1, :], temp['data'][1, :]
            print(f"filename: {file}")
            tPrefactor = (1/np.inf) + 1
            ax.loglog(ts/tPrefactor, mLT, color='k',label='SSRW fixed')  # prefactor is (1/inf) + 1 = 1 so ts/prefactor = ts
            ax2.loglog(ts, LMELT, color='k', label="SSRW fixed")
            ax3.loglog(ts / tPrefactor, LMELTS, color = 'k', label="SSRW fixed")
            ax4.loglog(ts, mLT, color='k', label="SSRW fixed")
    nSys = temp['n']
    tMax = temp['tMax']
    # legends and titles etc
    ax.legend(loc='lower right',fontsize='x-small')
    ax2.legend(loc='upper left',fontsize='x-small')
    ax3.legend(loc='upper left',fontsize='x-small')
    ax4.legend(loc='lower right',fontsize='x-small')
    ax.set_title(f"n={nSys}, v={v}, tMax={tMax}")
    ax2.set_title(f"n={nSys}, v={v}, tMax={tMax}")
    ax3.set_title(f"n={nSys}, v={v}, tMax={tMax}")
    ax4.set_title(f"n={nSys},v={v},tMax={tMax}")
    fig.savefig(os.path.join(topDir, "scaledMeanLTWithTimeScalingFixedWithSqrt2.png"),)
    fig2.savefig(os.path.join(topDir, "logMeanExpLTFixed.png"))
    fig3.savefig(os.path.join(topDir, "scaledLogMeanExpLTWithTimeScaledFixedWithSqrt2.png"))
    fig4.savefig(os.path.join(topDir,"scaledMeanLTFixedWithSqrt2.png"))
    plt.close('all')
    return
