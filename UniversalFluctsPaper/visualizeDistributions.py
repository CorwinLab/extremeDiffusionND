import numpy as np
from matplotlib import pyplot as plt
from randNumberGeneration import getRandomDistribution


def visualizeAllDistributions(n):
    distNames = ['Dirichlet']*7 + ['LogNormal','Delta','Corner']
    alphas = np.array([(np.sqrt(10)**i)/np.sqrt(1000) for i in range(7)])
    params = [[alpha]*4 for alpha in alphas] + [[0, 1]] + [""] + [""]
    labels = ['Dirichlet'+str(np.round(alpha,2)) for alpha in alphas] + ['LogNormal0, 1'] + ["Delta"] + ["Corner"]
    colors = plt.cm.tab10(np.linspace(0,1,len(distNames)))
    plt.ion()
    fig, ax = plt.subplots()
    tempfig, tempax = plt.subplots()

    for i in range(len(params)):
        func = getRandomDistribution(distNames[i],params[i])
        vals = np.array([func() for _ in range(n)])  # n by 4
        # if distNames[i] == 'LogNormal':
        #     sumVals = np.sum(vals, axis=1)
        #     normalizedVals = vals.T / sumVals
        #     vals = normalizedVals  # always in shape (n, 4)
        vals = vals[:,0]  # just take 1 direction.
        # #bins = np.unique(np.hstack([np.sort(vals)[::3], np.max(vals)]))
        # height, bins, _ = tempax.hist(vals, label=labels[i],bins=50,alpha=0.5,color=colors[i])
        # ax.plot(bins[1:], height, label=labels[i],color=colors[i])
        height, bins, _ = tempax.hist(vals,density=True,bins=50,label=labels[i],alpha=0.5,color=colors[i])
        ax.semilogy(bins[1:], height, label=labels[i],color=colors[i])
    ax.legend()
    ax.set_title("Choices of distribution")
    fig.savefig("distributions.png")

    # size: (# alphas, # samples, 4)
#    dirichlets = np.array([np.random.dirichlet([alpha]*4,size=(10000)) for alpha in alphas])
    # sum_dirichlets = np.sum(dirichlets,axis=2  # these are all just 1 because the sum of the 4 dirichlet nums is 1
    # now we just want 1 direction
    # plt.ion()
    # plt.figure()
    # for i in range(alphas.shape[0]):
    #     plt.hist(dirichlets[i,:,0],label=f"{alphas[i]: .3f}",alpha=0.5)

def visualizeCorner(n):
    colors = plt.cm.tab10(np.linspace(0,1,4))
    labels = ["-x","-y","y","x"]
    plt.ion()
    fig, ax = plt.subplots()
    tempfig, tempax = plt.subplots()
    func = getRandomDistribution('Corner',"")
    vals = np.array([func() for _ in range(n)])
    for i in range(len(labels)):
        height, bins, _ = tempax.hist(vals[:,i],label=labels[i],color=colors[i],alpha=0.5)
        ax.plot(bins[:-1],height,label=labels[i],color=colors[i])
    ax.legend()
    ax.set_title("corner distribution")
    fig.savefig("cornerDistVisualization.png")