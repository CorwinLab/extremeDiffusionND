import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt("./MatlabWeights/Weights.txt", delimiter=',')
print(data)
fig, ax = plt.subplots()
ax.plot(np.arange(0, len(data))[data!=0], data[data!=0])
fig.savefig("MatlabWeights.png", bbox_inches='tight')