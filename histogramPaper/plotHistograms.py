import numpy as np
import npquad
from matplotlib import pyplot as plt
import h5py
import os
import json

def gaussianPDF(x, mean, var):
    """ return gaussian pdf """
    return 1 / (np.sqrt(2 * np.pi * np.sqrt(var)) * np.exp(-1/2 * (x - mean)**2) / var)

def dirichletBinning(data, dirichletN, minWidth=0.05):
    """
    combination of dirichlet binning and other stuff?
    Parameters
    ----------
    data: h5py file, list of final proabilities
    dirichletN: number of dirichlet  bins
    minWidth: minimum width of bins

    Returns
    ---------
    binned data
    """
    data = np.sort(data)  # sort data
    bins = data[::dirichletN]  # put sorted data into dirichletN segments
    # group neighboring bins together if their width is smaller than minWidth
    lowEdge = bins[0]
    newBins = [lowEdge]
    for edge in bins[:-1]:
        if (edge - lowEdge) > minWidth:
            newBins.append(edge)
            lowEdge = edge
    newBins.append(bins[-1])
    return newBins

if __name__ == "__main__":
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    # setup gaussian stuff, demeaned so mu=0, sigma=1
    xvals = np.linspace(-5,5,num=1000)
    mean = 0
    var = 1

    dir = "/home/fransces/Documents/code/extremeDiffusionND/histogramPaper/largeVelocityAnalysis/multipleFinalProbs/"
    probsFile = os.path.join(dir,"FinalProbs.h5")


