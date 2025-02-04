from lDiffusionLink import libDiffusion
import numpy as np

class DiffusionND(libDiffusion.DiffusionND):
    def __init__(self, alphas, size):
        super().__init__(alphas, size)

    @property
    def alpha(self):
        return np.array(self.getAlpha())
    
    def generateBiases(self):
        return np.array(self.getRandomNumbers())
    
    @property
    def PDF(self):
        return np.array(self.getPDF())
    
    @PDF.setter
    def PDF(self, _PDF):
        self.setPDF(_PDF)
    
    @property
    def time(self):
        return self.getTime()
    
    @property
    def L(self):
        return self.getL()
    
    def integratedProbability(self, radii):
        return np.array(super().integratedProbability(radii))
    
    def logIntegratedProbability(self, radii):
        return np.array(super().logIntegratedProbability(radii))

if __name__ == '__main__':
    L = 3
    d = DiffusionND([1, 1, 1, 1], 3)
    time =  50
    xvals = np.arange(-L, L+1)
    xx, yy = np.meshgrid(xvals, xvals)
    distArray = np.sqrt(xx ** 2 + yy ** 2)
    
    for t in range(time):
        R = 1
        d.iterateTimestep()
        print(d.logIntegratedProbability([[R]]))
        