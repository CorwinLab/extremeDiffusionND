import numpy as np 
import os 
from matplotlib import pyplot as plt
import sys
import matplotlib.colors as colors
sys.path.append("../../../")
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
ax.set_xlim([10**-9, 10**-1])
ax.set_ylim([10**-10, 10**0])
ax.set_ylabel(r"$\mathrm{Var}(\ln(\mathrm{prob}))$")
ax.set_xlabel(r"$ \frac{\ln(t)}{4\pi t} v^2 \frac{\mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}{1 - \mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}$")

for distName in os.listdir(dir):
    if 'alpha' in distName:
        alpha_val = distName.replace("alpha", "")
        data_file = os.path.join('../Data/', f'{alpha_val}stats.npz')
        info_file = os.path.join(dir, distName, 'L5000', 'tMax10000', 'info.npz')

    else: 
        data_file = os.path.join('../Data/', f'{distName}stats.npz')
        info_file = os.path.join(dir, distName)
        params = os.listdir(info_file)[0]
        info_file = os.path.join(info_file, params, 'info.npz')

    if not os.path.exists(data_file):
        continue

    data = np.load(data_file)
    info = np.load(info_file)
    
    time = info['times']
    vs = info['velocities'][0]
    velocity_data = data['variance'][1, :, :] 
    
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
    
    prefactor = vs[::-1]**2 * expVar / (1 - expVar) / 4 / np.pi * np.log(time[-1]) / time[-1]
    
    ax.scatter(prefactor, velocity_data[-1, :][::-1], marker=vars['ms'][distName], edgecolors='k', lw=0.1)

xvals = np.geomspace(1e-9, 15)
ax.plot(xvals, xvals, ls='--', c='k')

fig.savefig("../Figures/DiffusiveNoiseVelocity.pdf", bbox_inches='tight')