import numpy as np
import os
import json
from matplotlib import pyplot as plt
import differenceRandomWalk as drw
from histogramPaper import plotLinePointStats as p
import time

# helper function to find the nearest time
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def calcAndSaveAllBetas(statsFileName):
    """
    returns list of betas, variances, measurement distances, times, and v=r/t values
    given a stats filename produced by geHistogramStats for past a Line
    """
    topDir = os.path.split(statsFileName)[0]
    with open(os.path.join(topDir,"variables.json"),"r") as v:
        variables = json.load(v)
    # tMax = variables['tMax']
    # times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    # if not time in times:
    #     time = find_nearest(times, time)  # use the closest value of allowed t to the requested one
    alpha = variables['alpha']
    processedStats = p.processLinePointStatsNPY(statsFileName)
    variances = processedStats[1,:]
    rs = processedStats[3,:]
    ts = processedStats[4,:]
    vs = rs / ts
    good = (vs < 1)
    variances = variances[good]
    rs = rs[good]
    ts = ts[good]
    vs = vs[good]
    betas = np.array([drw.computeBeta(alpha, v) for v in vs])
    saveFile = os.path.join(topDir,"allLineStatsWithBeta.npy")
    np.save(saveFile, np.array([betas, variances, rs, ts, vs]))
    return betas, variances, rs, ts, vs

def main():
    alpha003Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha003/LineStats.npy"
    alpha01Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha01/LineStats.npy"
    alpha1Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha1/LineStats.npy"

    tMax = 1000  # im doing a dumb and hardcoding this but whatever

    data003 = p.processLinePointStatsNPY(alpha003Path)
    variances003, rs003, ts003 = data003[1,:], data003[3,:], data003[4,:]
    vs003 = rs003/ts003

    data01 = p.processLinePointStatsNPY(alpha01Path)
    variances01, rs01, ts01 = data01[1,:],data01[3,:],data01[4,:]
    vs01 = rs01/ts01
    #
    data1 = p.processLinePointStatsNPY(alpha1Path)
    variances1, rs1, ts1 = data1[1,:],data1[3,:],data1[4,:]
    vs1 = rs1/ts1
    # this should be identical for any set of vs and ts
    # if we want we can set a different tMax to get it at different times
    good = (vs003 <=1 ) & (ts003 == tMax)
    bad = (vs003[good] < 1e-1)  # in theory this can be used to mask out the diffusive regime
    betas003 = np.array([drw.computeBeta(0.03,v) for v in vs003[good]])
    betas01 = np.array([drw.computeBeta(0.1,v) for v in vs01[good]])
    betas1 = np.array([drw.computeBeta(0.1,v) for v in vs1[good]])
    scaledVar003 = variances003[good] / betas003**2
    scaledVar01 = variances01[good] / betas01**2
    scaledVar1 = variances1[good] / betas1**2


    # plotting
    fig, ax = plt.subplots()
    ax.set_title(f"lnP past a line, tMax={tMax}, v>1e-1 \n scaled to be thru 0 and 1")
    ax.loglog(betas003[~bad], ((scaledVar003-np.min(scaledVar003))/(np.max(scaledVar003) - np.min(scaledVar003)))[~bad], '.', label="alpha=0.03", color='darkblue')
    ax.loglog(betas01[~bad],((scaledVar01-np.min(scaledVar01))/(np.max(scaledVar01) - np.min(scaledVar01)))[~bad], '.', label="alpha=0.1", color='darkgoldenrod')
    ax.loglog(betas1[~bad], ((scaledVar1-np.min(scaledVar1))/(np.max(scaledVar1) - np.min(scaledVar1)))[~bad], '.', label="alpha=1", color='darkgreen')
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\frac{1}{\beta^2}\mathrm{Var}[\ln{P_{line}}]$")
    ax.set_yscale('linear')
    ax.legend()
    plt.show()
    # #  here's what we'd do if we had the "bad" mask
    # plt.loglog(betas[~bad], variances[~bad] / (betas[~bad]) ** 2, '.', color='darkblue', label=f"alpha=0.03 at t=1000")
    # plt.loglog(betas2[~bad], variances2[~bad] / (betas2[~bad] ** 2), '.', color='darkgoldenrod',
    #            label=f"alpha=0.1 at t=1000")
    # plt.loglog(betas3[~bad], variances3[~bad] / (betas3[~bad] ** 2), '.', color='darkgreen', label=f"alpha=1 at t=1000")


def compareToDPRM(path, alpha):
    # alpha003Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha003/LineStats.npy"
    #alpha01Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha01/LineStats.npy"
    # alpha1Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha1/LineStats.npy"
    # path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha001/LineStats.npy"
    tMaxList = [3,10,31,100,316,1000]  # im doing a dumb and hardcoding this but whatever

    data = p.processLinePointStatsNPY(path)
    variances, rs, ts = data[1,:], data[3,:], data[4,:]
    mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
    variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
    vs = rs/ts
    # this should be identical for any set of vs and ts
    # if we want we can set a different tMax to get it at different times
    fig, ax = plt.subplots()
    fig1,ax1 = plt.subplots()
    ax.set_title(f"scaled lnP past line for alpha={alpha}, excluding v>=1 and v<1e-1 \n alpha=0.1")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\frac{1}{\beta^2}\mathrm{Var}[\ln{P_{line}}]$")
    for t in tMaxList:
        # print(f"t: {t}")
        good = (vs < 1) & (ts == t)
        variances, vs = variances[good], vs[good]
        _, unique_indices = np.unique(vs,return_index=True)
        variances, vs = variances[unique_indices], vs[unique_indices]
        betas = np.array([drw.computeBeta(alpha,v) for v in vs])
        temp = variances/betas**2
        bad = (vs < 1e-1)
        temp, betas = temp[~bad], betas[~bad]
        # print(f"shape of betas after ~bad: {betas.shape}")
        scaledTemp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        # beta vs scaled variance, set to be between 0 and 1
        ax.semilogx(betas, scaledTemp, '.-',label=f"t={t}")
        # beta vs variance
        ax1.loglog(betas, variances[~bad], '.-', label=f"t={t}")

        # reset for next run?
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
        variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
        vs = rs / ts
    ax.legend()
    ax1.legend()
    ax1.set_title(f"alpha={alpha},excluding v>=1 and v<1e-1")
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel(r"$\mathrm{Var}[\ln{P_{line}}]$")
    fig.show()
    fig1.show()
    return

def compareAlphas(t=1000):
    path0001 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha0.001/LineStats.npy"
    path0003 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha0.003/LineStats.npy"
    path001 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha001/LineStats.npy"
    path003 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha003/LineStats.npy"
    path01 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha01/LineStats.npy"
    path1 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha1/LineStats.npy"
    path10 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha10/LineStats.npy"
    path31 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha31/LineStats.npy"

    paths = [path0001, path0003, path001, path003, path01, path1, path10, path31]
    alphas = [0.001,0.003, 0.01, 0.03, 0.1, 1, 10, 31]
    fig, ax = plt.subplots()
    ax.set_title(f"scaled lnP past line at t=1000 for many alphas, excluding v>=1 and v<1e-1 \n alpha=0.1"
                 f"\n f(alpha)=a^2/(a0(a0+1))")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$f(alpha)\frac{1}{\beta^2}\mathrm{Var}[\ln{P_{line}}]$")

    fig1,ax1 = plt.subplots()
    ax1.set_title(f"t=1000 excluding v>=1 and v<1e-1 \n f(alpha)=a^2/(a0(a0+1))")
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel(r"$f(\alpha)\mathrm{Var}[\ln{P_{line}}]$")

    fig2, ax2, = plt.subplots()
    ax2.set_title(f"var[lnP] without v cutoff, unscaled \n f(alpha)=a(a+1)/(a0(a0+1))")
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel(r"$f(\alpha)\frac{1}{\beta^2}\mathrm{Var}[\ln{P_{line}}]$")

    fig3, ax3 = plt.subplots()
    ax3.set_title("t=1000 excluding v>=1, unscaled var \n f(alpha)=a^2/(a0(a0+1))")
    ax3.set_xlabel(r"$\beta$")
    ax3.set_ylabel(r"$f(\alpha)\mathrm{Var}[\ln{P_{line}}]$")
    for path in paths:
        # load data, name variables
        alpha = alphas[paths.index(path)]
        print(f"alpha: {alpha}")
        prefactor = alpha**2 / (4*alpha*(4*alpha+1))
        # prefactor = alpha
        data = p.processLinePointStatsNPY(path)
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
        variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
        vs = rs / ts
        # pull out the shit we want
        good = (vs < 1) & (ts == t)
        variances, vs = variances[good], vs[good]
        _, unique_indices = np.unique(vs,return_index=True)
        variances, vs = variances[unique_indices], vs[unique_indices]
        betas = np.array([drw.computeBeta(alpha,v) for v in vs])
        temp = variances/betas**2
        ax2.loglog(betas, temp*prefactor,'.-',label=f"alpha={alpha}")
        ax3.loglog(betas, variances+prefactor, '.-',label=f"alpha={alpha}")
        bad = (vs < 1e-1)
        temp, betas = temp[~bad]*prefactor, betas[~bad]
        # print(f"shape of betas after ~bad: {betas.shape}")
        scaledTemp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        # beta vs scaled variance, set to be between 0 and 1
        ax.semilogx(betas, scaledTemp, '.-',label=f"alpha={alpha}")
        # beta vs variance
        ax1.loglog(betas, variances[~bad]*prefactor, '.-', label=f"alpha={alpha}")

        # reset for next run?
        # variances, rs, ts = data[1, :], data[3, :], data[4, :]
        # mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
        # variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
        # vs = rs / ts
    ax.legend()
    # ax.set_yscale('log')
    fig.show()
    ax1.legend()
    fig1.show()
    ax2.legend()
    #ax2.set_xlim([1e-2,100])
    #ax2.set_ylim([1,10**4])
    fig2.show()
    ax3.legend()
    fig3.show()
    return