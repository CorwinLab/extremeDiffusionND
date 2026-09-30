import numpy as np
import os
import json
from matplotlib import pyplot as plt
import differenceRandomWalk as drw
from histogramPaper import plotLinePointStats as p
import time
import matplotlib.patches as mpatches

def calcAndSaveAllBetas(statsFileName):
    """
    returns list of betas, variances, measurement distances, times, and v=r/t values
    given a stats filename produced by geHistogramStats for past a Line
    """
    topDir = os.path.split(statsFileName)[0]
    with open(os.path.join(topDir, "variables.json"), "r") as v:
        variables = json.load(v)
    # tMax = variables['tMax']
    # times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    # if not time in times:
    #     time = find_nearest(times, time)  # use the closest value of allowed t to the requested one
    alpha = variables['alpha']
    processedStats = p.processLinePointStatsNPY(statsFileName)
    variances = processedStats[1, :]
    rs = processedStats[3, :]
    ts = processedStats[4, :]
    vs = rs / ts
    good = (vs < 1)
    variances = variances[good]
    rs = rs[good]
    ts = ts[good]
    vs = vs[good]
    betas = np.array([drw.computeBeta(alpha, v) for v in vs])
    saveFile = os.path.join(topDir, "allLineStatsWithBeta.npy")
    np.save(saveFile, np.array([betas, variances, rs, ts, vs]))
    return betas, variances, rs, ts, vs


def collapseTime():
    alpha0001Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha0.001/LineStats.npy"
    alpha003Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha003/LineStats.npy"
    alpha01Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha01/LineStats.npy"
    alpha1Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha1/LineStats.npy"
    alpha10Path = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha10/LineStats.npy"

    tMax = 1000  # im doing a dumb and hardcoding this but whatever
    data0001 = p.processLinePointStatsNPY(alpha0001Path)
    variances0001, rs0001, ts0001 = data0001[1, :], data0001[3, :], data0001[4, :]
    vs0001 = rs0001 / ts0001

    data003 = p.processLinePointStatsNPY(alpha003Path)
    variances003, rs003, ts003 = data003[1, :], data003[3, :], data003[4, :]
    vs003 = rs003 / ts003

    data01 = p.processLinePointStatsNPY(alpha01Path)
    variances01, rs01, ts01 = data01[1, :], data01[3, :], data01[4, :]
    vs01 = rs01 / ts01
    #
    data1 = p.processLinePointStatsNPY(alpha1Path)
    variances1, rs1, ts1 = data1[1, :], data1[3, :], data1[4, :]
    vs1 = rs1 / ts1

    data10 = p.processLinePointStatsNPY(alpha10Path)
    variances10, rs10, ts10 = data10[1, :], data1[3, :], data1[4, :]
    vs10 = rs10 / ts10
    # this should be identical for any set of vs and ts
    # if we want we can set a different tMax to get it at different times
    good = (vs003 <= 1) & (ts003 == tMax)
    bad = (vs003[good] < 1e-1)  # in theory this can be used to mask out the diffusive regime
    betas0001 = np.array([drw.computeBeta(0.001, v) for v in vs0001[good]])
    betas003 = np.array([drw.computeBeta(0.03, v) for v in vs003[good]])
    betas01 = np.array([drw.computeBeta(0.1, v) for v in vs01[good]])
    betas1 = np.array([drw.computeBeta(1, v) for v in vs1[good]])
    betas10 = np.array([drw.computeBeta(10, v) for v in vs10[good]])

    scaledVar0001 = variances0001[good] / betas0001 ** 2
    scaledVar003 = variances003[good] / betas003 ** 2
    scaledVar01 = variances01[good] / betas01 ** 2
    scaledVar1 = variances1[good] / betas1 ** 2
    scaledVar10 = variances10[good] / betas10 ** 2

    # plotting
    fig, ax = plt.subplots()
    ax.set_title(f"lnP past a line tMax={tMax} v>1e-1 \n scaled to be thru 0 and 1 by max/min stuff")
    ax.loglog(betas0001[~bad],
              ((scaledVar0001 - np.min(scaledVar0001)) / (np.max(scaledVar0001) - np.min(scaledVar0001)))[~bad], '.',
              label="alpha=0.001", color='orangered')
    ax.loglog(betas003[~bad],
              ((scaledVar003 - np.min(scaledVar003)) / (np.max(scaledVar003) - np.min(scaledVar003)))[~bad], '.',
              label="alpha=0.03", color='darkblue')
    ax.loglog(betas01[~bad], ((scaledVar01 - np.min(scaledVar01)) / (np.max(scaledVar01) - np.min(scaledVar01)))[~bad],
              '.', label="alpha=0.1", color='darkgoldenrod')
    ax.loglog(betas1[~bad], ((scaledVar1 - np.min(scaledVar1)) / (np.max(scaledVar1) - np.min(scaledVar1)))[~bad], '.',
              label="alpha=1", color='darkgreen')
    ax.loglog(betas10[~bad], ((scaledVar10 - np.min(scaledVar10)) / (np.max(scaledVar10) - np.min(scaledVar10)))[~bad],
              '.', label="alpha=10", color="mediumvioletred")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$\frac{1}{\beta^2}\mathrm{Var}[\ln{P_{line}}]$")
    ax.set_yscale('linear')
    ax.legend()
    fig.show()


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
    tMaxList = [3, 10, 31, 100, 316, 1000]  # im doing a dumb and hardcoding this but whatever

    data = p.processLinePointStatsNPY(path)
    variances, rs, ts = data[1, :], data[3, :], data[4, :]
    mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
    variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
    vs = rs / ts
    # this should be identical for any set of vs and ts
    # if we want we can set a different tMax to get it at different times
    # fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    # ax.set_title(f"scaled lnP past line for alpha={alpha}, excluding v>=1 and v<1e-1 \n alpha=0.1")
    # ax.set_xlabel(r"$\beta$")
    # ax.set_ylabel(r"$\frac{1}{\beta^2}\mathrm{Var}[\ln{P_{line}}]$")
    for t in tMaxList:
        # print(f"t: {t}")
        good = (vs < 1) & (ts == t)
        variances, vs = variances[good], vs[good]
        _, unique_indices = np.unique(vs, return_index=True)
        variances, vs = variances[unique_indices], vs[unique_indices]
        betas = np.array([drw.computeBeta(alpha, v) for v in vs])
        temp = variances / betas ** 2
        bad = (vs < 1e-1)
        temp, betas = temp[~bad], betas[~bad]
        # print(f"shape of betas after ~bad: {betas.shape}")
        # scaledTemp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        # beta vs scaled variance, set to be between 0 and 1
        # ax.semilogx(betas, scaledTemp, '.-',label=f"t={t}")
        # beta vs variance
        ax1.loglog(betas, variances[~bad], '.-', label=f"t={t}")

        # reset for next run?
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
        variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
        vs = rs / ts
    # ax.legend()
    ax1.legend()
    ax1.set_title(f"alpha={alpha},excluding v>=1 and v<1e-1")
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel(r"$\mathrm{Var}[\ln{P_{line}}]$")
    # fig.show()
    fig1.show()
    return


def compareAlphas(t=1000):
    # path0001 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha0.001/LineStats.npy"
    # path0003 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha0.003/LineStats.npy"
    # path001 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha001/LineStats.npy"
    # path003 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha003/LineStats.npy"
    # path01 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha01/LineStats.npy"
    # path1 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha1/LineStats.npy"
    # path10 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha10/LineStats.npy"
    # path31 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha31/LineStats.npy"
    path0001 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha0.001/logStatsP.npy"
    path0003 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha0.003/logStatsP.npy"
    path001 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha001/logStatsP.npy"
    path003 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha003/logStatsP.npy"
    path01 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha01/logStatsP.npy"
    path1 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha1/logStatsP.npy"
    path10 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha10/logStatsP.npy"
    path31 = "/home/fransces/Documents/code/extremeDiffusionND/pastLine/alpha31/logStatsP.npy"
    paths = [path0001, path0003, path001, path003, path01, path1, path10, path31]
    alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 1, 10, 31]
    fig, ax = plt.subplots()
    ax.set_title(f"log(varP) at t=1000 for many alphas, excluding v>=1")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$-\frac{1}{\beta^2}\ln{\mathrm{Var}[P_{line}]}$")
    #
    fig1, ax1 = plt.subplots()
    ax1.set_title(f"t=1000 excluding v>=1 and v<1e-1")
    ax1.set_xlabel(r"$\beta$")
    ax1.set_ylabel(r"$-\frac{1}{\beta^2}\ln{\mathrm{Var}[P_{line}]}$")
    #
    # fig2, ax2, = plt.subplots()
    # ax2.set_title(f"var[lnP] without v cutoff, unscaled \n f(alpha)=a(a+1)/(a0(a0+1))")
    # ax2.set_xlabel(r"$\beta$")
    # ax2.set_ylabel(r"$f(\alpha)\frac{1}{\beta^2}\mathrm{Var}[\ln{P_{line}}]$")
    #
    # fig3, ax3 = plt.subplots()
    # ax3.set_title("t=1000 excluding v>=1, unscaled var \n f(alpha)=a^2/(a0(a0+1))")
    # ax3.set_xlabel(r"$\beta$")
    # ax3.set_ylabel(r"$f(\alpha)\mathrm{Var}[\ln{P_{line}}]$")
    for path in paths:
        # load data, name variables
        alpha = alphas[paths.index(path)]
        print(f"alpha: {alpha}")
        # prefactor = alpha**2 / (4*alpha*(4*alpha+1))
        # prefactor = alpha
        data = p.processLinePointStatsNPY(path)
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
        variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
        vs = rs / ts
        # pull out the shit we want
        good = (vs < 1) & (ts == t)
        variances, vs = variances[good], vs[good]
        _, unique_indices = np.unique(vs, return_index=True)
        variances, vs = variances[unique_indices], vs[unique_indices]
        betas = np.array([drw.computeBeta(alpha, v) for v in vs])
        temp = variances / betas ** 2
        ax.loglog(betas, -temp, '.-', label=f"alpha={alpha}")
        # ax2.loglog(betas, temp*prefactor,'.-',label=f"alpha={alpha}")
        #ax3.loglog(betas, variances+prefactor, '.-',label=f"alpha={alpha}")
        bad = (vs < 1e-1)
        temp, betas = temp[~bad], betas[~bad]
        # print(f"shape of betas after ~bad: {betas.shape}")
        # scaledTemp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
        # beta vs scaled variance, set to be between 0 and 1
        #ax.semilogx(betas, scaledTemp, '.-',label=f"alpha={alpha}")
        # beta vs variance
        ax1.loglog(betas, -temp, '.-', label=f"alpha={alpha}")

        # reset for next run?
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        mask1 = (np.isfinite(rs))  # set of rs should also be constant so i dont need to redo it
        variances, rs, ts = variances[mask1], rs[mask1].astype(int), ts[mask1]
        vs = rs / ts
    ax.legend()
    # ax.set_yscale('log')
    fig.show()
    ax1.legend()
    fig1.show()
    # ax2.legend()
    # #ax2.set_xlim([1e-2,100])
    # #ax2.set_ylim([1,10**4])
    # fig2.show()
    # ax3.legend()
    # fig3.show()
    return


def varVsBeta(path):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    with open(os.path.join(path, "variables.json"), "r") as v:
        variables = json.load(v)
    tMax = variables['tMax']
    alpha = variables['alpha']
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    subTimes = times[10:-1:10]
    n = len(subTimes)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    print(f"alpha: {alpha} \n times: {subTimes}")
    # data = p.processLinePointStatsNPY(os.path.join(path,"logStatsP.npy"))
    data = p.processLinePointStatsNPY(os.path.join(path, "statsLogP.npy"))
    variances, rs, ts = data[1, :], data[3, :], data[4, :]
    finite = np.isfinite(rs)
    variances, rs, ts = variances[finite], rs[finite].astype(int), ts[finite]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.set_xlabel(r"$\beta$")
    # ax.set_ylabel(r"$-t\ln{\mathrm{Var}[P_{line}]}$ at t")
    ax.set_ylabel(r"$\frac{1}{t^{1/2}}\mathrm{Var}[\ln{P_{line}}]$")
    ax.set_title(f"alpha={alpha}")
    for i in range(len(subTimes)):
        t = subTimes[i]
        print(f"t: {t}")
        vs = rs / ts
        good = (vs < 1) & (ts == t)
        print(f"# of good datapoints: {np.sum(good)}")
        variances, rs, vs, ts = variances[good], rs[good], vs[good], ts[good]
        _, unique = np.unique(rs, return_index=True)
        variances, rs, vs, ts = variances[unique], rs[unique], vs[unique], ts[unique]
        print(f"# of unique rs: {len(unique)}")
        betas = np.array([drw.computeBeta(alpha, v) for v in vs])
        ax.loglog(betas, variances / np.sqrt(ts), '.-', lw=1, label=f"t={t}", color=colors[i])
        # reset for next loop
        # data = p.processLinePointStatsNPY(os.path.join(path, "logStatsP.npy"))
        data = p.processLinePointStatsNPY(os.path.join(path, "statsLogP.npy"))
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        finite = np.isfinite(rs)
        variances, rs, ts = variances[finite], rs[finite].astype(int), ts[finite]
    # fig.legend()
    fig.show()
    return


def varVsG(path):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    with open(os.path.join(path, "variables.json"), "r") as v:
        variables = json.load(v)
    tMax = variables['tMax']
    alpha = variables['alpha']
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    subTimes = times[10:-1:10]
    n = len(subTimes)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    print(f"alpha: {alpha} \n times: {subTimes}")
    data = p.processLinePointStatsNPY(os.path.join(path, "logStatsP.npy"))
    # data = p.processLinePointStatsNPY(os.path.join(path,"statsLogP.npy"))
    variances, rs, ts = data[1, :], data[3, :], data[4, :]
    finite = np.isfinite(rs)
    variances, rs, ts = variances[finite], rs[finite].astype(int), ts[finite]
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.set_xlabel(r"$g(2\mathrm{arctanh}(v))$")
    ax.set_ylabel(r"$-\frac{1}{t^{1/2}}\ln{\mathrm{Var}[P_{line}]}$ at t")
    # ax.set_ylabel(r"$t \mathrm{Var}[\ln{P_{line}}]$")
    ax.set_title(f"alpha={alpha}")
    for i in range(len(subTimes)):
        t = subTimes[i]
        print(f"t: {t}")
        vs = rs / ts
        good = (vs < 1) & (ts == t)
        print(f"# of good datapoints: {np.sum(good)}")
        variances, rs, vs, ts = variances[good], rs[good], vs[good], ts[good]
        _, unique = np.unique(rs, return_index=True)
        variances, rs, vs, ts = variances[unique], rs[unique], vs[unique], ts[unique]
        print(f"# of unique rs: {len(unique)}")
        Gs = np.array([drw.computeGNumeric(alpha, v) for v in vs])
        # betas = np.array([drw.computeBeta(alpha,v) for v in vs])
        ax.loglog(Gs, -variances / np.sqrt(ts), '.-', lw=1, label=f"t={t}", color=colors[i])
        # reset for next loop
        data = p.processLinePointStatsNPY(os.path.join(path, "logStatsP.npy"))
        # data = p.processLinePointStatsNPY(os.path.join(path, "statsLogP.npy"))
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        finite = np.isfinite(rs)
        variances, rs, ts = variances[finite], rs[finite].astype(int), ts[finite]
    # fig.legend()
    fig.show()
    return


def compareVGBeta(path):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})

    with open(os.path.join(path, "variables.json"), "r") as v:
        variables = json.load(v)
    tMax = variables['tMax']
    alpha = variables['alpha']
    times = np.unique(np.geomspace(1, tMax, 500).astype(int))
    subTimes = times[10:-1:10]
    n = len(subTimes)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    print(f"alpha: {alpha} \n times: {subTimes}")
    data = p.processLinePointStatsNPY(os.path.join(path, "logStatsP.npy"))
    fig, ax = plt.subplots(3,1,figsize=(6, 15), constrained_layout=True)
    fig.suptitle(f"alpha={alpha}")
    variances, rs, ts = data[1, :], data[3, :], data[4, :]
    finite = np.isfinite(rs)
    variances, rs, ts = variances[finite], rs[finite].astype(int), ts[finite]
    ax[0].set_xlabel(r"$v$")
    ax[0].set_ylabel(r"$-t\ln{\mathrm{Var}[P_{line}]}$ at t")
    ax[1].set_xlabel(r"$g(2\mathrm{arctanh}(v))$")
    ax[1].set_ylabel(r"$-t\ln{\mathrm{Var}[P_{line}]}$ at t")
    ax[2].set_xlabel(r"$\beta$")
    ax[2].set_ylabel(r"$-t\ln{\mathrm{Var}[P_{line}]}$ at t")
    for i in range(len(subTimes)):
        t = subTimes[i]
        # print(f"t: {t}")
        vs = rs / ts
        good = (vs < 1) & (ts == t)
        # print(f"# of good datapoints: {np.sum(good)}")
        variances, rs, vs, ts = variances[good], rs[good], vs[good], ts[good]
        _, unique = np.unique(rs, return_index=True)
        variances, rs, vs, ts = variances[unique], rs[unique], vs[unique], ts[unique]
        print(f"# of unique rs: {len(unique)}")
        Gs = np.array([drw.computeGNumeric(alpha, v) for v in vs])
        betas = np.array([drw.computeBeta(alpha, v) for v in vs])
        ax[0].loglog(vs, -variances * ts, '.-', lw=1, ms=2, color=colors[i])
        ax[1].loglog(Gs, -variances * ts, '.-', lw=1, ms=2, color=colors[i])
        ax[2].loglog(betas, -variances * ts, '.-', lw=1, ms=2, color=colors[i])
        # reset for next loop
        data = p.processLinePointStatsNPY(os.path.join(path, "logStatsP.npy"))
        # data = p.processLinePointStatsNPY(os.path.join(path, "statsLogP.npy"))
        variances, rs, ts = data[1, :], data[3, :], data[4, :]
        finite = np.isfinite(rs)
        variances, rs, ts = variances[finite], rs[finite].astype(int), ts[finite]
    # fig.legend()
    fig.show()
    fig.savefig(os.path.join(path,"compareVGBeta.png"))
    return

def gAlpha0(v):
    return np.log((1+v**2)**2)

def collapseVGBeta(pathList,savePath,tMax=1000):
    plt.rcParams.update(
        {'font.size': 15, 'text.usetex': True, 'text.latex.preamble': r'\usepackage{amsfonts, amsmath, bm}'})
    fig, ax = plt.subplots(3, 1, figsize=(6, 15), constrained_layout=True)
    ax[0].set_xlabel(r"$v$")
    ax[0].set_ylabel(r"$-\ln{\mathrm{Var}[P_{line}]}$")
    ax[1].set_xlabel(r"$g(2\mathrm{arctanh}(v))$")
    ax[1].set_ylabel(r"$-\ln{\mathrm{Var}[P_{line}]}$")
    ax[2].set_xlabel(r"$\beta$")
    ax[2].set_ylabel(r"$-\ln{\mathrm{Var}[P_{line}]}$")
    fig.suptitle(f"t=1000")
    fig2, ax2 = plt.subplots(3, 1, figsize=(6, 15), constrained_layout=True)
    ax2[0].set_xlabel(r"$v$")
    ax2[0].set_ylabel(r"$-\ln{\mathrm{Mean}[P_{line}]}$")
    ax2[1].set_xlabel(r"$g(2\mathrm{arctanh}(v))$")
    ax2[1].set_ylabel(r"$-\ln{\mathrm{Mean}[P_{line}]}$")
    ax2[2].set_xlabel(r"$\beta$")
    ax2[2].set_ylabel(r"$-\ln{\mathrm{Mean}[P_{line}]}$")
    labels = []
    n = len(pathList)
    colors = plt.cm.cool(np.linspace(0, 1, n))
    patchObjects = []
    for path in pathList:
        with open(os.path.join(path, "variables.json"), "r") as v:
            variables = json.load(v)
        tMax = variables['tMax']
        alpha = variables['alpha']
        labels.append(f"{alpha}")
        print(f"alpha: {alpha} @ tMax; {tMax}")
        data = p.processLinePointStatsNPY(os.path.join(path, "logStatsP.npy"))
        means, variances, rs, ts = data[0,:], data[1, :], data[3, :], data[4, :]
        finite = np.isfinite(rs)
        means, variances, rs, ts = means[finite], variances[finite], rs[finite].astype(int), ts[finite]
        # print(f"t: {t}")
        vs = rs / ts
        good = (vs < 1) & (ts == tMax)
        print(f"# of good datapoints: {np.sum(good)}")
        means, variances, rs, vs, ts = means[good], variances[good], rs[good], vs[good], ts[good]
        _, unique = np.unique(rs, return_index=True)
        means, variances, rs, vs, ts = means[unique], variances[unique], rs[unique], vs[unique], ts[unique]
        print(f"# of unique rs: {len(unique)}")
        Gs = np.array([drw.computeGNumeric(alpha, v) for v in vs])
        betas = np.array([drw.computeBeta(alpha, v) for v in vs])
        # variances
        ax[0].loglog(vs, -variances, '.-', lw=1, ms=2, color=colors[pathList.index(path)])
        ax[1].loglog(Gs, -variances, '.-', lw=1, ms=2, color=colors[pathList.index(path)])
        ax[2].loglog(betas, -variances, '.-', lw=1, ms=2, color=colors[pathList.index(path)])
        # means
        ax2[0].loglog(vs, -means, '.-', lw=1, ms=2, color=colors[pathList.index(path)])
        ax2[1].loglog(Gs, -means, '.-', lw=1, ms=2, color=colors[pathList.index(path)])
        ax2[2].loglog(betas, -means, '.-', lw=1, ms=2, color=colors[pathList.index(path)])
        patchObjects.append(mpatches.Patch(color=colors[pathList.index(path)],label=f"{alpha}"))
    # now do the totally sticky case
    tsData = np.load("/home/fransces/Documents/code/extremeDiffusionND/pastLine/totallySticky/radiiProbArray.npy")
    tsRs = tsData[0,:].astype(int)
    tsMean = tsData[1,:]
    tsVar = tsData[2,:]
    print('len of rSq: ', len(tsRs))
    print('loaded totally sticky')
    #tsMean = np.mean(tsProbArray, axis=0)
    #tsVar = np.var(tsProbArray, axis=0)
    tsVs = tsRs / tMax
    tsGood = (tsVs < 1)
    print(len(tsGood))
    tsMean, tsVar, tsRs, tsVs = tsMean[tsGood], tsVar[tsGood], tsRs[tsGood], tsVs[tsGood]
    _, unique = np.unique(tsRs, return_index=True)
    tsMean, tsVar, tsRs, tsVs = tsMean[unique], tsVar[unique], tsRs[unique], tsVs[unique]
    #tsGs = np.array([gAlpha0(v) for v in tsVs])
    tsGs = np.array([drw.computeBeta(0,v) for v in tsVs])
    patchObjects.append(mpatches.Patch(color='k',label="0"))
    labels.append(f"{0}")
    ax[0].loglog(tsVs, -np.log(tsVar),'.-',lw=1,ms=2, color='k')
    ax[1].loglog(tsGs, -np.log(tsVar),'.-',lw=1,ms=2, color='k')
    print('plotted means')
    ax2[0].loglog(tsVs, -np.log(tsMean),'.-',lw=1,ms=2, color='k')
    ax2[1].loglog(tsGs, -np.log(tsMean),'.-',lw=1,ms=2, color='k')
    print('plotted vars')
    fig.legend(patchObjects,labels,loc='lower right')
    fig.show()
    fig.savefig(os.path.join(savePath,"collapsedVGBetaAlpha.png"))
    fig2.legend(patchObjects,labels,loc='lower right')
    fig2.show()
    fig2.savefig(os.path.join(savePath,"collapsedMeanVGBetaAlpha.png"))
    return