import numpy as np
from matplotlib import pyplot as plt
import h5py
import os
import json

meanVarFile = os.path.join('/home/jacob/Desktop/', 'MeanVar.h5')

vars = os.path.join('/home/jacob/Desktop/', 'variables.json')

with h5py.File(meanVarFile, 'r') as f:
    numFiles = f.attrs['numberOfFiles']
    skew = f['linear']['skew'][:]

with open(vars, 'r') as f:
    vars = json.load(f)
    time = np.array(vars['ts'])
    velocity = np.array(vars['velocities'])

# Get corner dist skewness
moments = np.loadtxt("./conerDist/Moments.txt")
firstMoment = moments[:, 0]
secondMoment = moments[:, 1]
thirdMoment = moments[:, 2]

sigma = np.sqrt(secondMoment - firstMoment**2)
cornerSkew = (thirdMoment - 3 * firstMoment * sigma**2 - firstMoment**3) / sigma**3
conerTime = np.arange(0, 1000, 1)
print(cornerSkew)


cmap = plt.get_cmap('seismic')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_ylim([-0.4, 0.7])
ax.set_xlim([4, 10**3])
ax.set_xlabel("t")
ax.set_ylabel("Skewness")

for i in range(skew.shape[1]):
    radius = velocity[i] * time 
    timeCut = time[radius > 2]
    skewI = skew[:, i]
    skewI = skewI[radius > 2]
    color = (velocity[i] - min(velocity)) / (max(velocity) - min(velocity))
    ax.plot(timeCut, skewI, label=i, color=cmap(color))

ax.plot(conerTime, cornerSkew,  c='g')

fig.savefig("Skew.png", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xlabel("v")
ax.set_ylabel("Skewness")

for i in range(skew.shape[1]):
    radius = velocity[i] * time 
    timeCut = time[radius > 2]
    skewI = skew[:, i]
    skewI = skewI[radius > 2]
    color = (velocity[i] - min(velocity)) / (max(velocity) - min(velocity))
    ax.scatter(velocity[i], skewI[-1], color=cmap(color))

# ax.legend()
fig.savefig("SkewFinal.png", bbox_inches='tight')