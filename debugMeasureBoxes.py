import numpy as np
import os

from numba.np.npdatetime import timedelta_maximum_impl
from orca.speech_generator import STATE
from scipy.ndimage import morphology as m
import csv
import npquad
import pandas as pd
import sys
import glob
import evolve2DLattice as ev


def test(tMax, L, R, vs, distribution, params, barrierScale, sphereSaveFile=None, statesPath=None):
    if sphereSaveFile is None:
        sphereSaveFile = os.path.join("debugMeasureBoxes",'sphere'+'.txt')
    os.makedirs("debugMeasureBoxes", exist_ok=True)
    os.makedirs(os.path.join("debugMeasureBoxes", "states"),exist_ok=True)
    # statesPath = os.path.join(baseDirectory,states,str(sysID)+'.txt')
    # initialize occ with absorbing boundary,
    occ = np.zeros((2 * L + 1, 2 * L + 1))
    occ[L, L] = 1
    absorbingBoundary = ev.prepareBoundary(L, R)
    # ts = np.unique(np.geomspace(1, tMax, num=500).astype(int))  # generate times
    ts = ev.getListOfTimes(1, tMax, 500)  # default num=500

    # check if savefiles exist first
    if os.path.exists(sphereSaveFile):
        data = pd.read_csv(sphereSaveFile)
        max_time = max(data['Time'].values)
        # if file exists and is finished, exit
        if max_time == ts[-2]:
            print(f"File Finished", flush=True)
            sys.exit()

    # Set up writer and write header if save file doesn't exist,
    with open(sphereSaveFile, 'w') as f_sphere:
        # create writers
        writer_sphere = csv.writer(f_sphere)  # prob outside sphere
        # write data (since we switched from 'a' to 'w' this is fine)
        writer_sphere.writerow(["Time", *vs])

        # generate data
        for t, occ in ev.evolve2DLattice(occ, tMax, distribution, params, True, boundary=absorbingBoundary):
            # Get probabilities inside sphere
            if t in ts:
                RsScale = eval(barrierScale)
                Rs = list(np.array(vs * RsScale))

                # grab indices for box, past lines, and outside sphere
                sphere_masks = [ev.getInsideSphereMask(occ, r) for r in Rs]  # outside sphere

                # get probabilities outside sphere
                probs = [1 - np.sum(occ[mask]) for mask in sphere_masks]
                writer_sphere.writerow([t, *probs])
                f_sphere.flush()

                statesPath = os.path.join("debugMeasureBoxes","states",str(t)+'.txt')
                np.savetxt(statesPath,occ)
            # TODO: add in something that saves the state to a file
        f_sphere.close()
