import npquad
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import h5py

t = 1000
alpha = 1
numSystems = 2000000

logProbSystems = np.zeros(numSystems)
firstMoment = np.zeros(t)
secondMoment = np.zeros(t)
thirdMoment = np.zeros(t)

for id in tqdm(range(numSystems)):
    initialDirichlet = np.random.dirichlet([1]*4).astype(np.quad)

    prob = np.zeros(t).astype(np.quad)

    for i in range(0, 4):
        dirichletRandom = np.random.dirichlet([1]*4, size=t-1)
        singleDirection = dirichletRandom[:, 0]
        
        # Need to insert first dirichlet variable which is correlated 
        singleDirection = np.insert(singleDirection, 0, initialDirichlet[i]).astype(np.quad)
        prob += np.cumprod(singleDirection)

    logProb = np.log(prob).astype(float)
    
    firstMoment += logProb
    secondMoment += logProb ** 2
    thirdMoment += logProb ** 3
    
    logProbSystems[id] = logProb[-1]

moments = np.array([firstMoment / numSystems, secondMoment / numSystems, thirdMoment / numSystems]).T
np.savetxt("Moments.txt", moments)
np.savetxt("FinalProbs.txt", logProbSystems)

