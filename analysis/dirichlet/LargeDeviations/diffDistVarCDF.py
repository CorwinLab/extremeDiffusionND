import numpy as np 
import os 
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

dir = '/mnt/talapasShared'

distName = 'LogNormal'
params = os.listdir(os.path.join(dir, distName))[0]

data_file = os.path.join('../Data/', f'{distName}stats.npz')
info_file = os.path.join(dir, distName, params, 'info.npz')

data = np.load(data_file)
info = np.load(info_file)

time = info['times']
velocities = info['velocities'][0]

velocity_data = data['mean'][0, :, :] 

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-5, 10**3])
ax.set_xlabel(r"$\mathrm{t}$")
ax.set_ylabel(r"$\mathrm{Mean}(\ln(\mathrm{prob}))$")

def theory(v, t):
    return v**2 * t

for i in range(velocity_data.shape[1]):
    lines = ax.plot(time[:velocity_data.shape[0]], -velocity_data[:, i])
    linecolor = lines[0].get_color()
    ax.plot(time[:velocity_data.shape[0]], theory(velocities[i], time[:velocity_data.shape[0]]), ls='--', c=linecolor)

fig.savefig(f"../Figures/LargeDeviationMean{distName}.pdf", bbox_inches='tight')

velocity_data = data['variance'][0, :, :] 

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-8, 100])
ax.set_xlabel(r"$\mathrm{t}$")
ax.set_ylabel(r"$\mathrm{Variance}(\ln(\mathrm{prob}))$")

for i in range(velocity_data.shape[1]):
    ax.plot(time[:velocity_data.shape[0]], velocity_data[:, i])
    # ax.plot(time[:velocity_data.shape[0]], theory(velocities[i], time[:velocity_data.shape[0]]), ls='--')

xvals = np.geomspace(10**2, 10**4)
ax.plot(xvals, np.log(xvals) / 1e4, ls='--', c='k') 
fig.savefig(f"../Figures/LargeDeviationVariance{distName}.pdf", bbox_inches='tight')