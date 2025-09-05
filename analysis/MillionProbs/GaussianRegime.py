import numpy as np
import npquad 
from matplotlib import pyplot as plt
import h5py 
import json
import os

directory = '/mnt/talapasShared/MillionQuad2DProbs'
regime = 'sqrt'

colors = ['tab:red', 'tab:blue']
fig, ax = plt.subplots()
ax.set_xlabel("Velocity")
ax.set_ylabel("Skew")

def varianceTheory(v, alpha, t):
    betaSquared = v**2  / 4 / np.pi / alpha * np.log(t) / t
    return np.log(1 / (1 - betaSquared))

for d in os.listdir(directory):
    file = os.path.join(directory, d, 'MeanVar.h5')
    variables = os.path.join(directory, d, 'variables.json')
    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            critical = f[regime]
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
        else:
            color = 'tab:green'

        for i in range(skew.shape[1]):
            ax.scatter(velocities[i], skew[-1, i], color = color)

fig.savefig(f"./Figures/{regime}Skew.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Velocity")
ax.set_ylabel("Skew")

for d in os.listdir(directory):
    file = os.path.join(directory, d, 'MeanVar.h5')
    variables = os.path.join(directory, d, 'variables.json')
    
    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            critical = f[regime]
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
        else:
            color = 'tab:green'

        for i in range(skew.shape[1]):
            ax.scatter(velocities[i], variance[-1, i], color = color)

        # ax.plot(velocities, varianceTheory(np.array(velocities), alphas[0]), c=color)

fig.savefig(f"./Figures/{regime}Var.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Velocity")
ax.set_ylabel("Skew")

d = '1,1,1,1'
file = os.path.join(directory, d, 'MeanVar.h5')
variables = os.path.join(directory, d, 'variables.json')
cm = plt.get_cmap("jet")

if os.path.exists(file):
    with h5py.File(file, 'r') as f:
        critical = f[regime]
        variance = critical['var'][:]

    with open(variables, 'r') as f:
        vars = json.load(f)
    
    alphas = np.array(vars['params'])
    velocities = np.array(vars['velocities'])
    times = np.array(vars['ts'])
    
    if alphas[0] == 1:
        color = 'tab:red'
    elif alphas[0] == 0.1:
        color = 'tab:blue'
    
    for i in range(variance.shape[1]):
        validTimes = ((velocities[i] * np.sqrt(times)) > 3)
        prefactor = velocities[i] ** 2 / alphas[0] / 4 / np.pi
        ax.plot(times[validTimes], variance[:, i][validTimes] / prefactor, color=cm(i / variance.shape[1]))

ax.plot(times[times > 10], np.log(times[times > 10]) / times[times > 10], ls='--', c='k')

fig.savefig(f"./Figures/{regime}VarTime.png", bbox_inches='tight')

fig, ax = plt.subplots()
finalProbs = os.path.join(directory, 'FinalProbs.h5')

with h5py.File(finalProbs) as f:
    f[regime]