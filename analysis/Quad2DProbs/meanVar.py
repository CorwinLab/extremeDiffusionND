import numpy as np
import npquad 
import glob
import glob 
import h5py
import numpy as np
from tqdm import tqdm

meanVarFile = h5py.File("MeanVar.h5", 'a')

files = glob.glob('/mnt/talapasShared/Quad2DProbs/1,1,1,1/*.h5')
maxTime = 1998
moments = ['mean', 'secondMoment', 'thirdMoment', 'fourthMoment', 'var', 'skew']

with h5py.File(files[0], 'r') as f:
    for regime in f['regimes'].keys():
        meanVarFile.require_group(regime)
        for moment in moments:
            meanVarFile[regime].require_dataset(moment, shape=f['regimes'][regime].shape, dtype=float)
            meanVarFile[regime][moment][:] = np.zeros(f['regimes'][regime].shape, dtype=float)

num_files = 0
for f in tqdm(files): 
    with h5py.File(f, 'r') as f: 
        if f.attrs['currentOccupancyTime'] < maxTime:
            continue

        for regime in f['regimes'].keys():
            probs = f['regimes'][regime][:]

            meanVarFile[regime]['mean'][:] += np.log(probs).astype(float)
            meanVarFile[regime]['secondMoment'][:] += (np.log(probs) ** 2).astype(float)
            meanVarFile[regime]['thirdMoment'][:] += (np.log(probs) ** 3).astype(float)
            meanVarFile[regime]['fourthMoment'][:] += (np.log(probs) ** 4).astype(float)

        num_files += 1

print("Number of files used:", num_files)
meanVarFile.attrs['NumberOfFiles'] = num_files
    
for regime in meanVarFile.keys():
    meanVarFile[regime]['mean'][:] /= num_files
    meanVarFile[regime]['secondMoment'][:] /= num_files
    meanVarFile[regime]['thirdMoment'][:] /= num_files 
    meanVarFile[regime]['fourthMoment'][:] /= num_files

    meanVarFile[regime]['var'][:] = meanVarFile[regime]['secondMoment'][:] - meanVarFile[regime]['mean'][:] ** 2
    meanVarFile[regime]['skew'][:] = (meanVarFile[regime]['thirdMoment'][:] - 3 * meanVarFile[regime]['mean'][:] * meanVarFile[regime]['var'][:] - meanVarFile[regime]['mean'][:]**3) / (meanVarFile[regime]['var'][:]) **(3/2)

meanVarFile.close()