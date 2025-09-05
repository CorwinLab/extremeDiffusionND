import numpy as np 
import os 
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import json

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

distName = 'alpha0.1'

dir = '/mnt/talapasShared/h5Data/dirichlet/ALPHA0.1/L5000/tMax10000/'

data_file = os.path.join(dir, 'Stats.h5')
# info_file = os.path.join(dir, distName, params, 'info.npz') # For LogNormal
info_file = os.path.join(dir, 'variables.json')

with h5py.File(data_file) as f:
    data = f['tOnSqrtLogT']
    mean = data['mean'][:]
    var = data['var'][:]
    skew = data['skew'][:]
    print(data.keys())
    
with open(info_file, 'r') as f:
    vars = json.load(f)
    time = np.array(vars['ts'])
    velocities = np.array(vars['velocities'])

maxVal = 0.5
cm = LinearSegmentedColormap.from_list('myCmap', [(0, 'r'), (maxVal / 2, 'b') , (1, 'r')])

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-5, 10**3])
ax.set_xlabel(r"$\mathrm{t}$")
ax.set_ylabel(r"$\mathrm{Mean}(\ln(\mathrm{prob}))$")

def theory(v, t):
    return v**2 * np.ones(len(t))

for i in range(mean.shape[1]):
    lines = ax.plot(time[:mean.shape[0]], -mean[:, i])
    linecolor = lines[0].get_color()
    ax.plot(time[:mean.shape[0]], theory(velocities[i], time[:mean.shape[0]]), ls='--', c=linecolor)

fig.savefig(f"./Figures/DiffusiveMean{distName}.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-8, 10])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Variance}\left(\ln\left(\mathbb{P}\left(\Vert S(t) \Vert > \frac{vt}{\sqrt{\ln(t)}} \right) \right)\right)$")

for i in range(var.shape[1]):
    goodTimes = velocities[i] * time / np.sqrt(np.log(time)) > 2
    ax.plot(time[goodTimes][:var.shape[0]], var[:, i][goodTimes], color = cm(i / var.shape[1]), label=velocities[i])

xvals = np.geomspace(10**2, 10**4)
fig.savefig(f"./Figures/DiffusiveVariance{distName}.pdf", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim([10**-1, 10])
ax.set_xlim([2, 10**4])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\frac{1}{v^2}\mathrm{Variance}\left(\ln\left(\mathbb{P}\left(\Vert S(t) \Vert > v \frac{t}{\sqrt{\ln(t)}} \right) \right)\right)$")

alphaFloat = float(distName.replace('alpha', ''))
prefactor = velocities**2

for i in range(var.shape[1]-4):
    goodTimes = velocities[i] * time / np.sqrt(np.log(time)) > 2
    newTimes = time[goodTimes][:var.shape[0]]
    newVar = var[:, i][goodTimes]

    if i == var.shape[1]-6:
        newTimes = newTimes[:-10]
        newVar = newVar[:-10]
    
    ax.plot(newTimes, newVar / prefactor[i], color = cm(i / var.shape[1]))

xvals = np.geomspace(10**2, 10**4)

fig.savefig(f"./Figures/DiffusiveVarianceCollapse{distName}.svg", bbox_inches='tight')

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_ylim([-2, 2])
ax.set_xlim([10, 10**4])
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\mathrm{Skew}\left(\ln\left(\mathbb{P}\left(\Vert S(t) \Vert > v \sqrt{t} \right) \right)\right)$")

alphaFloat = float(distName.replace('alpha', ''))
prefactor = velocities**2

for i in range(skew.shape[1]):
    goodTimes = velocities[i] * time / np.sqrt(np.log(time)) > 2
    newTime = time[goodTimes][:skew.shape[0]]
    newSkew = skew[:, i][goodTimes] / prefactor[i]
    
    avgTime, avgSkew = log_moving_average(newTime, newSkew, 10**(1/20))

    ax.plot(avgTime, avgSkew, color = cm(i / skew.shape[1]))

xvals = np.geomspace(10**2, 10**4)
ax.plot(xvals, np.log(xvals) / xvals / 100, ls='--', c='k', label=r'$\frac{\ln(t)}{t}$')
leg = ax.legend(
	loc="upper right",
	framealpha=0,
	labelcolor=['k'],
	handlelength=0,
	handletextpad=0,
)
for item in leg.legendHandles:
	item.set_visible(False)
     
# for i, text in enumerate(leg.get_texts()):
# 	  plt.setp(text, color = colors[i])
        
fig.savefig(f"./Figures/DiffusiveSkew{distName}.svg", bbox_inches='tight')