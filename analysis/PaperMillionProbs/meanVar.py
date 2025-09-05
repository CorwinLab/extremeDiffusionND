import glob
import numpy as np
import npquad
import h5py
from tqdm import tqdm
import os

dir = '/mnt/corwinLab/2DData/PaperMillionQuad2DProbs'

meanVarFile = h5py.File(os.path.join(dir, "MeanVar.h5"), 'a')
finalProbs = h5py.File(os.path.join(dir, "FinalProbs.h5"), 'a')

fileNums = 5_000_000

maxTime = 999
moments = ['mean', 'secondMoment', 'thirdMoment', 'var', 'skew']

firstFileName = os.path.join(dir, "0.h5")
with h5py.File(firstFileName, 'r') as f:
    for regime in f['regimes'].keys():
        meanVarFile.require_group(regime)
        finalProbs.require_dataset(f'temp{regime}', shape=(fileNums, f['regimes'][regime].shape[1]), dtype=float)

        for moment in moments:
            meanVarFile[regime].require_dataset(moment, shape=f['regimes'][regime].shape, dtype=float)
            meanVarFile[regime][moment][:] = np.zeros(f['regimes'][regime].shape, dtype=float)

num_files = 0
for fileID in tqdm(range(fileNums)):
    fileName = os.path.join(dir, f'{fileID}.h5')
    try:
        with h5py.File(fileName, 'r') as f:
            if f.attrs['currentOccupancyTime'] < maxTime:
                continue

            for regime in f['regimes'].keys():
                probs = f['regimes'][regime][:]

                finalProbs[f'temp{regime}'][num_files, :] = np.log(probs[-1, :]).astype(float)

                with np.errstate(divide='ignore'):
                    meanVarFile[regime]['mean'][:] += (np.log(probs)).astype(float)
                    meanVarFile[regime]['secondMoment'][:] += (np.log(probs) ** 2).astype(float)
                    meanVarFile[regime]['thirdMoment'][:] += (np.log(probs) ** 3).astype(float)

            num_files += 1
    except Exception as e:
        continue

meanVarFile.attrs['numberOfFiles'] = num_files
print(num_files)

for regime in meanVarFile.keys():
    meanVarFile[regime]['mean'][:] /= num_files
    meanVarFile[regime]['secondMoment'][:] /= num_files
    meanVarFile[regime]['thirdMoment'][:] /= num_files

    with np.errstate(invalid='ignore', divide='ignore'):
        meanVarFile[regime]['var'][:] = meanVarFile[regime]['secondMoment'][:] - meanVarFile[regime]['mean'][:] ** 2
        sigma = np.sqrt(meanVarFile[regime]['var'][:])
        meanVarFile[regime]['skew'][:] = (meanVarFile[regime]['thirdMoment'][:] - 3 * meanVarFile[regime]['mean'][:] * sigma ** 2 - meanVarFile[regime]['mean'][:] ** 3) / sigma ** 3

    nonzeroProbs = finalProbs[f'temp{regime}'][:num_files, :]
    finalProbs.create_dataset(regime, data=nonzeroProbs)
    del finalProbs[f'temp{regime}']

meanVarFile.close()
finalProbs.close()
