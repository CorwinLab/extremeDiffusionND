import numpy as np 
import os 
from matplotlib import pyplot as plt
import sys
import matplotlib.colors as colors
import h5py
import json
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
ax.set_xlim([5 * 10**-9, 5 * 10**-1])
ax.set_ylim([10**-9, 10])
ax.set_ylabel(r"$\mathrm{Var}(\ln(\mathrm{prob}))$")
ax.set_xlabel(r"$ \frac{1}{4\pi} v^2 \frac{\mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}{1 - \mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}$")

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
    
    prefactor = vs[::-1]**2 * expVar / (1 - expVar) / 4 / np.pi
    
    ax.scatter(prefactor, velocity_data[-1, :][::-1], marker=vars['ms'][distName], edgecolors='k', lw=0.1)

# Plot the new data that I got
for alpha in [0.1, 1]:
    file = f'/mnt/talapasShared/MillionQuad2DProbs/{alpha},{alpha},{alpha},{alpha}/MeanVar.h5'
    variables = f'/mnt/talapasShared/MillionQuad2DProbs/{alpha},{alpha},{alpha},{alpha}/variables.json'

    with h5py.File(file, 'r') as f:
        critical = f['tOnSqrtLogT']
        variance = critical['var'][:]

    with open(variables, 'r') as f:
        vars = json.load(f)

    velocities = np.array(vars['velocities'])
    times = vars['ts']

    expVar = 1 / (1 + 4 * alpha)

    prefactor = velocities **2 * expVar / (1 - expVar) / 4 / np.pi
    ax.scatter(prefactor, variance[-1, :], marker='*', edgecolors='k', lw=0.1)

xvals = np.geomspace(1e-9, 15)
ax.plot(xvals, np.log(1 / (1 - xvals)), ls='--', c='k')

fig.savefig("./Figures/NoiseVelocity.pdf", bbox_inches='tight')