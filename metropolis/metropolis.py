import numpy as np
from tqdm import tqdm
from scipy.linalg import eigvalsh

def pij(xij):
    return np.exp(-xij**2 / 2)

def computeBin(lambdaC, binLambdaMin, binLambdaMax, nBins):
    if lambdaC > binLambdaMax:
        return nBins - 1
    elif lambdaC < binLambdaMin:
        return 0
    else:
        return int((lambdaC - binLambdaMin) / (binLambdaMax - binLambdaMin) * (nBins - 1))

def metropolisEigenValues(outSteps, inSteps, N, f = np.e, b = 0.92, nBins=100, width=0.1, binLambdaMin=1, binLambdaMax=15):
    #Let's say that weights runs from -10 to 10 with 100 bins
    logWeights = np.zeros(nBins)
    histogram = np.zeros(nBins)
    nc = 0
    tensor = np.random.randn(N,N)
    # Symmetrize Matrix
    tensor += tensor.T
    lambdaMax = np.max(np.linalg.eigvalsh(tensor))
    accepted = 0
    rejected = 0
    for _ in tqdm(range(outSteps)):
        for _ in range(inSteps):
            # Pick a random site
            i,j = np.sort(np.random.randint(0,N, size=2))
            proposedJump = np.random.randn()*width
            newTensor = tensor.copy()
            newTensor[i,j] += proposedJump
            newTensor[j,i] += proposedJump
            lambdaMaxNew = np.max(np.linalg.eigvalsh(newTensor))
            binLambdaNew = computeBin(lambdaMaxNew, binLambdaMin, binLambdaMax, nBins)
            binLambda = computeBin(lambdaMax, binLambdaMin, binLambdaMax, nBins)
            monteCarloProbability = (pij(newTensor[i,j]) / pij(tensor[i,j])) * np.exp(logWeights[binLambdaNew] - (logWeights[binLambda]))
            
            if np.isnan(monteCarloProbability):
                raise ValueError
            
            if np.random.rand() < monteCarloProbability:
                lambdaMax = lambdaMaxNew
                tensor = newTensor.copy()
                accepted += 1
            else:
                rejected += 1
            
            binLambda = computeBin(lambdaMax, binLambdaMin, binLambdaMax, nBins)
            logWeights[binLambda] -= np.log(f)
            histogram[binLambda] += 1
        
        # This resets the histogram if it is "flat" and reduces
        # the modification factor f
        choppedHistogram = histogram[1:-1]
        nonZeroChopped = choppedHistogram[choppedHistogram != 0]
        
        if np.min(nonZeroChopped) > np.mean(nonZeroChopped) * b:
            f = np.sqrt(f)
            histogram *= 0
            nc += 1

    return logWeights, nc, f, histogram, accepted, rejected


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    outSteps = 1000
    inSteps = 100
    N = 10
    weights, nc, f, histogram, accepted, rejected = metropolisEigenValues(outSteps, inSteps, N, binLambdaMin=1, binLambdaMax=5)
    print("Nc: ", nc)
    
    weights = weights[1:-1]
    histogram = histogram[1:-1]
    histogram = np.loadtxt("./Data/Histogram.txt")
    weights = np.loadtxt("./Data/Weights.txt")

    xvals = np.linspace(1, 15, 100)[1:-1]
    
    def weightsToDist(weights, xvals):
        weights = -weights
        weights = weights - np.max(weights)
        binWidth = np.diff(xvals)[0]
        
        dist = np.exp(weights)
        norm = np.sum(dist * binWidth)
        dist /= norm 

        return dist
    
    dist = weightsToDist(weights, xvals)

    fig, ax = plt.subplots()
    ax.plot(xvals, histogram)
    fig.savefig("./Figures/Histogram1.png", bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(xvals, weights - np.max(weights))
    fig.savefig("./Figures/Weights1.png", bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(xvals, histogram / np.exp(weights - np.max(weights)))
    fig.savefig("./Figures/DOS.png")

    eigvals = np.loadtxt("./Data/GOEEigenvalues.txt")

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.plot(xvals, dist)
    ax.hist(eigvals, density=True, bins='fd')
    fig.savefig("./Figures/Distribution.png")

        