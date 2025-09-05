import glob 
import h5py
import numpy as np
from tqdm import tqdm

finalProbs = h5py.File("FinalProbs132.h5", 'a')

files = glob.glob('/mnt/corwinLab/1,1,1,1/*.h5')
totalFiles = len(files)

maxTime = 9998

with h5py.File(files[0], 'r') as f:
    for regime in f['regimes'].keys():
        finalProbs.require_dataset(f'temp{regime}', shape=(totalFiles, f['regimes'][regime].shape[1]), dtype=float)

num_files = 0
for f in tqdm(files):
    with h5py.File(f, 'r') as f: 
        if f.attrs['currentOccupancyTime'] < maxTime:
            continue

        for regime in f['regimes'].keys():
            probs = f['regimes'][regime][:].astype(float)
            
            finalProbs[f'temp{regime}'][num_files, :] = probs[-1, :]
        
        num_files += 1

for regime in finalProbs.keys():
    nonzeroProbs = finalProbs[regime][:num_files, :]
    finalProbs.create_dataset(regime.replace("temp", ''), data=nonzeroProbs)
    del finalProbs[regime]

finalProbs.close()