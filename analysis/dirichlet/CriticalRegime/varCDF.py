import numpy as np 
import os 
from matplotlib import pyplot as plt
import sys
import matplotlib.colors as colors
sys.path.append("../../")
from memEfficientEvolve2DLattice import getExpVarX

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

vars = {'ms': {'Dirichlet': 'o',
               'Delta': '^',
               'LogNormal': 's',
               'LogUniform': 'd',
               }
               }

dir = '/mnt/talapasShared'

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$\alpha \mathrm{Var}(\ln(\mathrm{prob}))$")

for distName in os.listdir(dir):
    if 'alpha' not in distName:
        continue

    alpha_val = distName.replace("alpha", "")
    data_file = os.path.join('./Data/', f'{alpha_val}stats.npz')
    info_file = os.path.join(dir, distName, 'L5000', 'tMax10000', 'info.npz')

    if not os.path.exists(data_file):
        continue

    data = np.load(data_file)
    info = np.load(info_file)
    
    time = info['times']
    vs = info['velocities'][0]
    velocity_data = data['variance'][3, :, :] 

    ax.plot(vs, velocity_data[-1, :] * float(alpha_val))

vvals = np.array([10**-3, 5*10**-1])
ax.plot(vvals, vvals**2 / 10, ls='--', c='k', label=r'$v^2$')
ax.legend()
fig.savefig("./Figures/Velocities.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\frac{1}{v^2} \mathrm{Var}(\ln(\mathrm{prob}))$")
ax.set_xlabel(r"$\mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)$")

cm1 = colors.LinearSegmentedColormap.from_list("MyCmap", ['tab:blue', 'tab:purple', 'tab:red'])

for distName in os.listdir(dir):
    if 'alpha' in distName:
        alpha_val = distName.replace("alpha", "")
        data_file = os.path.join('./Data/', f'{alpha_val}stats.npz')
        info_file = os.path.join(dir, distName, 'L5000', 'tMax10000', 'info.npz')

    else: 
        data_file = os.path.join('./Data/', f'{distName}stats.npz')
        info_file = os.path.join(dir, distName)
        params = os.listdir(info_file)[0]
        info_file = os.path.join(info_file, params, 'info.npz')

    if not os.path.exists(data_file):
        continue

    data = np.load(data_file)
    info = np.load(info_file)
    
    time = info['times']
    vs = info['velocities'][0]
    velocity_data = data['variance'][3, :, :] 
    
    if 'alpha' in distName:
        expVar = 1 / (1 + 4 * float(alpha_val))
        distName = 'Dirichlet'
    else: 
        if params != 'None':
            paramsVals = params.split(",")
            paramsVals = np.array(paramsVals).astype(float)
        else:
            paramsVals = ''
        
        expVar = getExpVarX(distName, paramsVals)

    im = ax.scatter(np.ones(len(vs)) * expVar, velocity_data[-1, :][::-1] / vs[::-1]**2, 
                    norm=colors.LogNorm(vmin=vs.min(), vmax=vs.max()), marker=vars['ms'][distName], c=vs[::-1], cmap=cm1, edgecolors='k', lw=0.1)

cbar = plt.colorbar(im, ax=ax)
cbar.ax.set_ylabel(r"$v$", rotation=270, labelpad=15)

xvals = np.geomspace(0.05, 0.9)
ax.plot(xvals, xvals / (1 - xvals) / 10 / 1.5, c='k', ls='--', label=r'$\frac{\mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}{1-\mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}$')
ax.legend()

fig.savefig("./Figures/Noise.pdf", bbox_inches='tight')