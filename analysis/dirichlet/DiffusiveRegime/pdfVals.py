import numpy as np 
from matplotlib import pyplot as plt
import os
from scipy.stats import skew, kurtosis

plt.rcParams.update({'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, color}'})

distName = 'LogNormal'

data = f'../Data/{distName}pdfVals.npz'
data = np.load(data)['pdfVals']
dir = '/mnt/talapasShared'
info_file = os.path.join(dir, distName, '0,1', 'info.npz')

# Pick out the vt / sqrt(log(t)) regime
critData = data[:, 1, :]

# Get velocities and times
info = np.load(info_file)
time = info['times']
vs = info['velocities'][0]

cutoff=5

normalizedPDFVals = []
skewness = []
kurtosisVals = []
for i in range(len(vs[:-cutoff])):
    pdfvals = critData[:, i]
    normalizedVals = (pdfvals - np.mean(pdfvals)) / np.sqrt(np.var(pdfvals))
    normalizedPDFVals.append(normalizedVals)
    skewness.append(skew(normalizedVals))
    kurtosisVals.append(kurtosis(normalizedVals))

normalizedPDFVals = np.array(normalizedPDFVals).flatten()
print(normalizedPDFVals.shape)
def gaussian(x, mean, var):
    return 1 / np.sqrt(2 * np.pi * var) * np.exp(-1/2 * (x- mean)**2 / var)

xvals = np.linspace(-4, 4, 500)
gausDist = gaussian(xvals, 0, 1)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_yscale("log")
ax.set_xlim([-6, 4])
ax.set_ylim([10**-3, 1])
ax.set_xlabel(r"$\frac{\ln(\mathrm{prob}) - \mu}{\sigma}$")
ax.set_ylabel(r"$\mathrm{Probability\ Density}$")
ax.hist(normalizedPDFVals, bins=100, density=True)
ax.plot(xvals, gausDist, ls='--', c='k')

ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])
ax2.scatter(vs[:-cutoff], skewness, c='b', s=15)
ax2.scatter(vs[:-cutoff], kurtosisVals, c='r', s=15)
ax2.hlines(0, min(vs[:-cutoff]), max(vs[:-cutoff]), ls='--', color='k')
ax2.set_xscale("log")
ax2.set_xlabel(r"$v$")
ax2.set_ylabel(r"$\textcolor{blue}{\mathrm{Skewness}}/\mathrm{Kurtosis}$", labelpad=-1)

fig.savefig("../Figures/DiffusiveHist.pdf", bbox_inches='tight')