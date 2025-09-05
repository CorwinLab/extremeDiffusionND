import numpy as np
import npquad 
from matplotlib import pyplot as plt
import h5py 
import json
import os
plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

def log_moving_average(time, data, window_size=10):
    assert window_size > 1
    window_min = time[0]
    window_max = window_size * window_min
    new_times = []
    mean_data = []
    while window_min <= max(time):
        window_time = time[(time >= window_min) & (time < window_max)]
        window_data = data[(time >= window_min) & (time < window_max)]

        if len(window_data) == 0: 
            mean_data.append(np.nan)
            new_times.append(np.exp(np.mean(np.log(window_time))))
        else:
            mean_data.append(np.mean(window_data))
            new_times.append(np.exp(np.mean(np.log(window_time))))

        window_min = window_max
        window_max = window_size * window_min

    return np.array(new_times), np.array(mean_data)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\mathrm{Var}(\ln(\mathrm{prob}))$")
ax.set_xlabel(r"$ \frac{1}{4\pi} v^2 \frac{\mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}{1 - \mathrm{Var}_{\nu}\left(\mathbb{E}^{\bm{\xi}}\left[\vec{X}\right]\right)}$")

directory = '/mnt/talapasShared/MillionQuad2DProbs/'
for d in os.listdir(directory):
    file = os.path.join(directory, d, 'MeanVar.h5')
    variables = os.path.join(directory, d, 'variables.json')

    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            critical = f['tOnSqrtLogT']
            var = critical['var'][:]

        with open(variables, 'r') as f:
            vars = json.load(f)

        alphas = np.array(vars['params'])
        velocities = np.array(vars['velocities'])
        times = np.array(vars['ts'])

        expVar = 1 / (1 + 4 * alphas[0])

        prefactor = velocities **2 * expVar / (1 - expVar) / 4 / np.pi
        ax.scatter(prefactor, var[-1, :], label=None)
    
xvals = np.array([10**-4 / 2, 10**-2])
ax.plot(xvals, xvals, ls='--', c='k', label=r'$x$')

xvals = np.array([5*10**-2, 2 * 10**-1])
ax.plot(xvals, xvals**2.5 * 100 * 1.5, ls='--', c='k', label=r'$x^{2.5}$')

ax.legend()

fig.savefig("./Figures/CriticalFinalVariance.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_ylabel(r"$\mathrm{Skew}(\ln(\mathrm{prob}))$")
ax.set_xlabel(r"$v$")

directory = '/mnt/talapasShared/MillionQuad2DProbs/'
for d in os.listdir(directory):
    file = os.path.join(directory, d, 'MeanVar.h5')
    variables = os.path.join(directory, d, 'variables.json')

    if os.path.exists(file):
        with h5py.File(file, 'r') as f:
            critical = f['tOnSqrtLogT']
            skew = critical['skew'][:]

        with open(variables, 'r') as f:
            vars = json.load(f)

        alphas = np.array(vars['params'])
        velocities = np.array(vars['velocities'])
        times = np.array(vars['ts'])

        expVar = 1 / (1 + 4 * alphas[0])
        if alphas[0] == 1:
            color = 'b'
        else:
            color = 'r'

        for i in range(skew.shape[1]):
            avgTime, avgSkew = log_moving_average(times, skew[:, i], 10**(1/10))
            ax.scatter(velocities[i], avgSkew[-1], color=color)

fig.savefig("./Figures/CriticalFinalSkew.pdf", bbox_inches='tight')