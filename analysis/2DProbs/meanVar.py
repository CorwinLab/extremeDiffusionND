import glob 
import h5py
import numpy as np
from tqdm import tqdm

meanVarFile = h5py.File("MeanVar.h5", 'a')
finalProbs = h5py.File("FinalProbs200.h5", 'a')

files = glob.glob('/mnt/talapasShared/2DProbs/Dirichlet/1,1,1,1/*.h5')
totalFiles = len(files)

maxTime = 9998
moments = ['mean', 'secondMoment', 'thirdMoment', 'var', 'skew']

with h5py.File(files[0], 'r') as f:
    for regime in f['regimes'].keys():
        meanVarFile.require_group(regime)
        finalProbs.require_dataset(f'temp{regime}', shape=(totalFiles, f['regimes'][regime].shape[1]), dtype=float)

        for moment in moments:
            meanVarFile[regime].require_dataset(moment, shape=f['regimes'][regime].shape, dtype=float)
            meanVarFile[regime][moment][:] = np.zeros(f['regimes'][regime].shape, dtype=float)

num_files = 0
for f in tqdm(files):
    with h5py.File(f, 'r') as f: 
        if f.attrs['currentOccupancyTime'] < maxTime:
            continue

        for regime in f['regimes'].keys():
            probs = f['regimes'][regime][:].astype(float)
            
            finalProbs[f'temp{regime}'][num_files, :] = probs[131, :]
            
            with np.errstate(divide='ignore'):
                meanVarFile[regime]['mean'][:] += np.log(probs)
                meanVarFile[regime]['secondMoment'][:] += np.log(probs) ** 2
                meanVarFile[regime]['thirdMoment'][:] += np.log(probs) ** 3

        num_files += 1

for regime in finalProbs.keys():
    meanVarFile[regime]['mean'][:] /= num_files
    meanVarFile[regime]['secondMoment'][:] /= num_files
    meanVarFile[regime]['thirdMoment'][:] /= num_files

    with np.errstate(invalid='ignore', divide='ignore'):
        meanVarFile[regime]['var'][:] = meanVarFile[regime]['secondMoment'][:] - meanVarFile[regime]['mean'][:] ** 2
        sigma = np.sqrt(meanVarFile[regime]['var'][:])
        meanVarFile[regime]['skew'][:] = (meanVarFile[regime]['thirdMoment'][:] - 3 * meanVarFile[regime]['mean'][:] * sigma ** 2 - meanVarFile[regime]['mean'][:] ** 3) / sigma ** 3

    nonzeroProbs = finalProbs[regime][:num_files, :]
    finalProbs.create_dataset(regime.replace("temp", ''), data=nonzeroProbs)
    del finalProbs[regime]

# meanVarFile.close()
finalProbs.close()