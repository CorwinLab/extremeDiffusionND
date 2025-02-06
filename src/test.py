import numpy as np
import npquad 
from libDiffusion import DiffusionND
import sys

L = 3
d = DiffusionND([1, 1, 1, 1], 3)
d.saveOccupancy("Occ.bin")
occ = d.loadOccupancy("Occ.bin", L)
print(occ)