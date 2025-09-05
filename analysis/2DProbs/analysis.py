import numpy as np
import h5py
from matplotlib import pyplot as plt
import json
from matplotlib import colors

regime = 'linear'
varsFile = '/mnt/corwinLab/1,1,1,1/variables.json'

with open(varsFile) as f:
    vars = json.load(f)
    time = np.array(vars['ts'])
    velocities = np.array(vars['velocities'])

cm = colors.LinearSegmentedColormap.from_list("myCmap", ['tab:blue', 'tab:red'])

with h5py.File("MeanVar.h5", 'r') as f:
    mean = f[regime]['mean'][:]
    var = f[regime]['var'][:]
    skew = f[regime]['skew'][:]

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1, 10**4])

    for i in range(mean.shape[1]):
        v = velocities[i]
        color = cm((np.log(v) - np.log(min(velocities))) / (np.log(max(velocities))-np.log(min(velocities))))
        ax.plot(time, -(mean[:, i]), color=color)
        ax.plot(time, v**2 * time / 4, color=color, ls='--')

    fig.savefig(f"./Figures/{regime}Mean.pdf", bbox_inches='tight')   

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([1, 10**4])
    
    for i in range(mean.shape[1]):
        ax.plot(time, var[:, i], color=cm((np.log(velocities[i]) - np.log(min(velocities))) / (np.log(max(velocities))-np.log(min(velocities)))), label=velocities[i])
        if np.all(np.isnan(var[:, i])): 
            continue
        else:
            idx = (~np.isnan(var[:, i])).cumsum().argmax()

    fig.savefig(f"./Figures/{regime}Var.pdf", bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_xlim([1, 10**4])
    ax.set_ylim([-1, 1])
    for i in range(mean.shape[1]):
        ax.plot(time, skew[:, i], color=cm((np.log(velocities[i]) - np.log(min(velocities))) / (np.log(max(velocities))-np.log(min(velocities)))), label=velocities[i])
        
    # ax.legend()
    fig.savefig(f"./Figures/{regime}Skew.pdf", bbox_inches='tight')