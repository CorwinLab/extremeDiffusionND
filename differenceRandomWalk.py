import numpy as np
import os
from time import time as wallTime
from numba import njit
import json
# import tracemalloc
from datetime import date
import sys
import shutil

# This module is intended to evolve the PMF of the difference random walk
# defined as \vec{V(t)} = \vec{R1(t)} - \vec{R2(t)}
# under the tilted two-point measure defined in Jacob's 2d Random Walk overleaf
# document, specifically eqns 19 and 20
# the goal is ultimately to be able to compute kappa, and thus the invariant
# measure mu, to get a scaling for the local time.

# don't worry about numba or logsumExp or storing things as logP yet
# assumes dirichlet,


# there should be 9 sites which each step takes. i will need to go out at least
# 2 lattice sites
# so i need to make my occupancy 4*tMax+1 by 4*tmax + 1, and then I need
# tMax of them
def initializeOccupancy(tMax):
    return np.zeros((4*tMax+1,4*tMax+1, tMax))

def normalization(alpha, v):
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
    normalizer = np.sum(terms)
    return normalizer

def calcPR1R2(r1,r2, alpha, v):
    """ this should return a proability for the change from r1 to r2"""
    diff = r2 - r1
    nhats = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
    a0 = 4 * alpha
    # sum n1 + n2 = r2 - r1 for n1 and n2 in set of nhats
    probSum = 0
    for n1 in nhats:
        for n2 in nhats:
            if n1 + n2 == diff:
                if r1 == 0:  # if the 2 walks are at the same site
                    # sameCov and diffCov are the E_nu[xi(n1)xi(n2)] values for
                    # a Dirichlet distributed thing with all equal alphas
                    sameCov = alpha * (alpha + 1) / (a0 * (a0 + 1))
                    diffCov = alpha ** 2 / (a0 * (a0 + 1))
                    if n1 == n2:
                        prefactor = sameCov
                    else:
                        prefactor = diffCov
                    probSum += prefactor * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[0])))
                else:  # if the 2 walks are separated
                    prefactor = ((1 - v**2)**2 ) / 16
                    probSum += prefactor * np.exp(2 * np.arctanh(v) * np.dot(nhats[0], (nhats[3] + nhats[0])))
    if r1 == 0:  # this is the 1/alpha^2 tetrm in eqn 19
        normalize = normalization(alpha,v)
        probSum *= normalize
    return probSum


def evolveDifferenceWalklattice(occupancy, time):
    origin = occupancy.shape[0] // 2