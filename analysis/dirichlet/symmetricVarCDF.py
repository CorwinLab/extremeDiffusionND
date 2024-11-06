import numpy as np 
import os 
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

dir = '/mnt/talapas/2DSqrtLogT/symmetricDirichlet/1,1,1,1'

data_file = os.path.join('./Data/', f'symmetricstats.npz')
info_file = os.path.join(dir, 'info.npz')

data = np.load(data_file)
info = np.load(info_file)
velocities = info['velocities']
time = info['times']
velocity_data = data['variance'][0, :, :] 

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\mathrm{t}$")
ax.set_ylabel(r"$\mathrm{Variance}(\ln(\mathrm{prob}))$")
ax.set_ylim([10**-9, 10])

for i in range(velocity_data.shape[1]):
    ax.plot(time[:velocity_data.shape[0]], velocity_data[:, i], label=velocities[0][i])
    
fig.savefig("./Figures/SymmetricVar.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$\mathrm{Var}(\ln(\mathrm{prob}))$")

time = info['times']
vs = info['velocities'][0]
velocity_data = data['variance'][0, :, :] 

ax.scatter(vs, velocity_data[-1, :])

vvals = np.array([10**-3, 5*10**-1])
ax.plot(vvals, vvals**2 / 10, ls='--', c='k', label=r'$v^2$')
ax.legend()
fig.savefig("./Figures/VelocitiesSymmetric.pdf", bbox_inches='tight')