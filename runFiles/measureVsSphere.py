import sys 
sys.path.append("../")
from evolve2DLattice import measureAtVsOnSphere
import numpy as np
import json 
import os

def saveVars(vars, save_file):
	"""
	Save experiment variables to a file along with date it was ran and
	"""
	for key, item in vars.items():
		if isinstance(item, np.ndarray):
			vars[key] = list(item)

	with open(save_file, "w+") as file:
		json.dump(vars, file)

if __name__ == '__main__': 
	# tMax, L, topDir, distribution, params, sysID = '1000', '100', '.', 'dirichlet', '4', '0'
	tMax, L, topDir, distribution, params, sysID = sys.argv[1:]

	L = int(L)
	tMax = int(float(tMax))
	sphereSaveFile = os.path.join(topDir, f"Sphere{sysID}.txt")
	lineSaveFile = os.path.join(topDir, f"Line{sysID}.txt")
	if distribution == 'dirichlet':
		params = float(params)
	else:
		params = None

	R = L - 1
	vs = np.unique(np.geomspace(1/100, 1, num=21))

	varsFile = os.path.join(topDir, 'variables.json')
	vars = {"tMax": tMax, 
		 	"L": L,
			"R": R,
			"vs": vs,
			"distribution": distribution,
			"params": params,
			"sphereSaveFile": sphereSaveFile, 
			"lineSaveFile": lineSaveFile}
	
	if int(sysID) == 0:
		saveVars(vars, varsFile)

	# measureAtVsOnSphere(tMax, L, R, vs , distribution, params, sphereSaveFile, lineSaveFile):
	measureAtVsOnSphere(tMax, L, R, vs, distribution, params, sphereSaveFile, lineSaveFile)