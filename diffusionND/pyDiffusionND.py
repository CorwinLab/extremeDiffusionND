from lDiffusionLink import libDiffusion
import numpy as np
import npquad

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
        return self.getPDF()
    
    @PDF.setter
    def PDF(self, _PDF):
        self.setPDF(_PDF)
    
    @property
    def time(self):
        return self.getTime()
    
    @time.setter 
    def time(self, _time):
        self.setTime(_time)
    
    @property
    def L(self):
        return self.getL()
    
    @classmethod
    def fromOccupancy(cls, alphas, size, occ, time):
        d = cls(alphas, size)
        d.PDF = occ
        d.time = time 
        return d
    
    def integratedProbability(self, radii):
        return np.array(super().integratedProbability(radii))
    
    def logIntegratedProbability(self, radii):
        return np.array(super().logIntegratedProbability(radii))
        