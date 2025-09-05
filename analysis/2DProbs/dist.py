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

with h5py.File("FinalProbs132.h5", 'r') as f:
    vals = f['linear'][:, 16]
    print(vals, np.min(vals), np.max(vals))
    mean = np.mean(vals)
    var = np.var(vals)
    normalized = (vals - mean) / np.sqrt(var)
    
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.hist(normalized, bins=100)
    fig.savefig("Figures/Dist.pdf", bbox_inches='tight')