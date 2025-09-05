from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({'font.size': 20, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts}'})

# Create colormap for skewness
colors = ['blue', 'black', 'goldenrod']
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

# Make full plot
fig, ax = plt.subplots()
time_steps = 4
step_size = 5
x = np.arange(-0.5, 11, 0.5)
y = np.arange(-0.5, 11, 0.5)

z = np.random.rand(x.shape[0]-1, y.shape[0]-1)

z[:, ::2] = np.nan
z[::2, :] = np.nan
z /= z 
z /= 2

def writerow(ax, xmin, xmax, y, zorder):
	xvals = np.arange(xmin, xmax, 0.5)
	yvals = np.tile(np.arange(-0.5 + y, 0.5 + y, 0.5).flatten(), xvals.shape[0] // 2)
	z = np.ones((xvals.shape[0]-1, yvals.shape[0]-1)) / 4
	z[::2, :] = np.nan
	z[:, ::2] = np.nan
	ax.pcolormesh(yvals, xvals, z, cmap='Blues', vmin = 0, vmax=1, zorder=zorder)

for i in range(11):
	zorder = 0 - 2 * i + 1
	writerow(ax, -0.5, 11.5, i + 0.5, zorder = zorder)

lattice_xvals = np.arange(-(step_size // 2), step_size // 2 + 1, 1)

dx = np.arange(-2, 3, 1)
dy_val = 1

xstart = 5.25 
ystart = 0.25 

exp_alpha = 1 
dir_alpha = 0.5

mutation_scale = 50

rand_vals = np.random.dirichlet([1, 1, 1, 1])
jumps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

# Add in dirichlet jumps
center = (2 + 0.25, 2 + 0.25)
for j in range(len(rand_vals)):
	jump = jumps[j]
	arrow = FancyArrowPatch((center[0], center[1]), (center[0] + jump[0], center[1] + jump[1]), mutation_scale = mutation_scale * (rand_vals[j] + 0.1), edgecolor='k', facecolor='k', alpha=exp_alpha, zorder=np.inf)
	ax.add_patch(arrow)

# Add in average arrow
# avgJumpx = np.sum(rand_vals * jumps[:, 0])
# avgJumpy = np.sum(rand_vals * jumps[:, 1])
# arrow = FancyArrowPatch((center[0], center[1]), (center[0] + avgJumpx, center[1] + avgJumpy), mutation_scale = mutation_scale * 0.75, edgecolor='r', facecolor='r', alpha=exp_alpha, zorder=np.inf)
# ax.add_patch(arrow)

ax.tick_params(left=False, labelleft=False)
ax.tick_params(bottom=False, labelbottom=False)

ax.set_ylim([0, 4.5])
ax.set_xlim([0, 4.5])
ax.set_ylabel(r"$y$")
ax.set_xlabel(r"$x$")
ax.set_title(r"$t=0$")
ax.set_aspect("equal")

fig.savefig("./Figures/EnvironmentMarchMeeting1.svg", bbox_inches='tight')

fig, ax = plt.subplots()
time_steps = 4
step_size = 5
x = np.arange(-0.5, 11, 0.5)
y = np.arange(-0.5, 11, 0.5)

z = np.random.rand(x.shape[0]-1, y.shape[0]-1)

z[:, ::2] = np.nan
z[::2, :] = np.nan
z /= z 
z /= 2

def writerow(ax, xmin, xmax, y, zorder):
	xvals = np.arange(xmin, xmax, 0.5)
	yvals = np.tile(np.arange(-0.5 + y, 0.5 + y, 0.5).flatten(), xvals.shape[0] // 2)
	z = np.ones((xvals.shape[0]-1, yvals.shape[0]-1)) / 4
	z[::2, :] = np.nan
	z[:, ::2] = np.nan
	ax.pcolormesh(yvals, xvals, z, cmap='Blues', vmin = 0, vmax=1, zorder=zorder)

for i in range(11):
	zorder = 0 - 2 * i + 1
	writerow(ax, -0.5, 11.5, i + 0.5, zorder = zorder)

lattice_xvals = np.arange(-(step_size // 2), step_size // 2 + 1, 1)

dx = np.arange(-2, 3, 1)
dy_val = 1

xstart = 5.25 
ystart = 0.25 

exp_alpha = 1 
dir_alpha = 0.5

mutation_scale = 50

jumps = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

# Add in dirichlet jumps
center = (2 + 0.25, 2 + 0.25)
for offset in jumps:
	for j in range(len(rand_vals)):
		rand_vals = np.random.dirichlet([1, 1, 1, 1])
		jump = jumps[j]
		arrow = FancyArrowPatch((center[0] + offset[0], center[1] + offset[1]), (center[0] + jump[0] + offset[0], center[1] + jump[1] + offset[1]), mutation_scale = mutation_scale * (rand_vals[j] + 0.1), edgecolor='k', facecolor='k', alpha=exp_alpha, zorder=np.inf)
		ax.add_patch(arrow)

ax.tick_params(left=False, labelleft=False)
ax.tick_params(bottom=False, labelbottom=False)

ax.set_ylim([0, 4.5])
ax.set_xlim([0, 4.5])
ax.set_ylabel(r"$y$")
ax.set_xlabel(r"$x$")
ax.set_title(r"$t=1$")
ax.set_aspect("equal")

fig.savefig("./Figures/EnvironmentMarchMeeting2.svg", bbox_inches='tight')