import numpy as np
import h5py
import sys
import os
import psutil

# Load new h5 file
fileName = "myFile.h5"
f = h5py.File(fileName, 'a')

# 825716
if 'test' in f.keys(): 
    rand_array = f['test'][:]
    print("Loaded from saved file")
else:
    print("initiating file")
    rand_array = np.zeros(shape=(10_000, 10_000))
    rand_array[5000, 5000] = 1
    f.create_dataset("test", data=rand_array)

while True:
    continue

f.close()