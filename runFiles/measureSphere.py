import sys 
sys.path.append("../")
from evolve2DLattice import measureOnSphere
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
	# tMax, L, topDir, sysID = '100', '10', '.', '0'
	tMax, L, topDir, sysID = sys.argv[1:]

	L = int(L)
	tMax = int(float(tMax))
	sphereSaveFile = os.path.join(topDir, f"Sphere{sysID}.txt")
	lineSaveFile = os.path.join(topDir, f"Line{sysID}.txt")
	
	R = L - 1
	Rs = np.unique(np.geomspace(2, R, num=20).astype(int))
	Rs = [int(r) for r in Rs]

	varsFile = os.path.join(topDir, 'variables.json')
	vars = {"tMax": tMax, 
		 	"L": L,
			"R": R,
			"Rs": Rs,
			"sphereSaveFile": sphereSaveFile, 
			"lineSaveFile": lineSaveFile}
	
	if int(sysID) == 0:
		saveVars(vars, varsFile)

	measureOnSphere(tMax, L, R, Rs, sphereSaveFile, lineSaveFile)