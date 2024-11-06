import numpy as np 
import os 
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

dir = '/mnt/talapasShared'

alpha = '0.1'
alpha_dir = f'alpha{alpha}'

data_file = os.path.join('./Data/', f'{alpha}stats.npz')
info_file = os.path.join(dir, alpha_dir, 'L5000', 'tMax10000', 'info.npz')

data = np.load(data_file)
info = np.load(info_file)
# print(info['velocities'])
time = info['times']
velocity_data = data['variance'][3, :, :] 

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-8, 400])
ax.set_xlabel(r"$\mathrm{t}$")
ax.set_ylabel(r"$\mathrm{Var}(\ln(\mathrm{prob}))$")

for i in range(velocity_data.shape[1]):
    ax.plot(time[:velocity_data.shape[0]], velocity_data[:, i])

fig.savefig("./Figures/Var.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$v$")
ax.set_ylabel(r"$\alpha \mathrm{Var}(\ln(\mathrm{prob}))$")

for alpha in os.listdir(dir):
    alpha_val = alpha.replace("alpha", "")
    data_file = os.path.join('./Data/', f'{alpha_val}stats.npz')
    info_file = os.path.join(dir, alpha, 'L5000', 'tMax10000', 'info.npz')

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
ax.set_xlabel(r"$\mathbb{E}^{\bm{\xi}}[\mathrm{Var}(\vec{X})]$")

for alpha in os.listdir(dir):

    alpha_val = alpha.replace("alpha", "")
    data_file = os.path.join('./Data/', f'{alpha_val}stats.npz')
    info_file = os.path.join(dir, alpha, 'L5000', 'tMax10000', 'info.npz')

    if not os.path.exists(data_file):
        continue

    data = np.load(data_file)
    info = np.load(info_file)
    
    time = info['times']
    vs = info['velocities'][0]
    velocity_data = data['variance'][3, :, :] 

    for i in range(velocity_data.shape[1]):
        if vs[i] > 10**-1:
            continue
        expVar = 1 / (1 + 4 * float(alpha_val))
        ax.scatter(expVar, velocity_data[-1, i] / vs[i]**2)

delta_dir = '/mnt/talapas/2DSqrtLogT/delta/None'
data_file = os.path.join("./Data/deltastats.npz")
info_file = os.path.join(delta_dir, 'info.npz')

data = np.load(data_file)
info = np.load(info_file)
time = info['times']
vs = info['velocities'][0]
velocity_data = data['variance'][0, :, :]

for i in range(velocity_data.shape[1]):
    if vs[i] > 10**-1:
        continue
    expVar = 1 / 3
    ax.scatter(expVar, velocity_data[-1, i] / vs[i]**2, marker='^')

xvals = np.geomspace(0.1, 0.9)
ax.plot(xvals, xvals / (1 - xvals) / 10 / 1.5, c='k', ls='--', label=r'$\frac{\mathbb{E}^{\bm{\xi}}[\mathrm{Var}(\vec{X})]}{1-\mathbb{E}^{\bm{\xi}}[\mathrm{Var}(\vec{X})]}$')
ax.legend()

fig.savefig("./Figures/Noise.pdf", bbox_inches='tight')