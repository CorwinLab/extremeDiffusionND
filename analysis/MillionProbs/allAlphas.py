import numpy as np
import npquad 
from matplotlib import pyplot as plt
import h5py 
import json
import os

directory = '/mnt/talapasShared/MillionQuad2DProbs'

colors = ['tab:red', 'tab:blue']
fig, ax = plt.subplots()
ax.set_xlabel("Velocity")
ax.set_ylabel("Skew")

for d in os.listdir(directory):
    file = os.path.join(directory, d, 'MeanVar.h5')
    variables = os.path.join(directory, d, 'variables.json')
    
    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            critical = f['linear']
            skew = critical['skew'][:]
            mean = critical['mean'][:]

        with open(variables, 'r') as f:
            vars = json.load(f)
        
        alphas = vars['params']
        velocities = vars['velocities']
        times = vars['ts']
        
        if alphas[0] == 1:
            color = 'tab:red'
        elif alphas[0] == 0.1:
            color = 'tab:blue'

        for i in range(skew.shape[1]):
            ax.scatter(velocities[i], skew[-1, i], color = color)

fig.savefig("./Figures/LinearkewVelocity.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xlabel("Velocity")
ax.set_ylabel("Skew")

for d in os.listdir(directory):
    file = os.path.join(directory, d, 'MeanVar.h5')
    variables = os.path.join(directory, d, 'variables.json')
    
    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            critical = f['linear']
            variance = critical['var'][:]

        with open(variables, 'r') as f:
            vars = json.load(f)
        
        alphas = vars['params']
        velocities = vars['velocities']
        times = vars['ts']
        
        if alphas[0] == 1:
            color = 'tab:red'
        elif alphas[0] == 0.1:
            color = 'tab:blue'

        for i in range(skew.shape[1]):
            ax.scatter(velocities[i], variance[-1, i], color = color)

def varianceTheory(v, alpha):
    betaSquared = v**2 / 4 / np.pi / alpha 
    print(betaSquared)
    return np.log(1 / (1 - betaSquared))

ax.plot(velocities, varianceTheory(np.array(velocities), 0.1))


fig.savefig("./Figures/VarianceVelocity.png", bbox_inches='tight')