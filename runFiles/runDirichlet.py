from diffusionND.runScripts import runDirichlet, getListOfTimes, saveVars, calculateRadii, linear, tOnSqrtLogT
import numpy as np
import sys
import os
from time import time as wallTime
from datetime import date

if __name__ == '__main__':
    # Test Code
	# L, tMax, distName, params, directory, systID = 1000, 50, 'Dirichlet', '1,1,1,1', './', 0
    L = int(sys.argv[1])
    tMax = int(sys.argv[2])
    distName = sys.argv[3]
    params = sys.argv[4]
    directory = sys.argv[5]
    systID = int(sys.argv[6])
    # Need to parse params into an array unless it is an empty string
    if params == 'None':
        params = ''
    else:
        params = params.split(",")
        params = np.array(params).astype(float)

    ts = getListOfTimes(tMax - 1, 1)
    velocities = np.concatenate((np.linspace(0.1, 0.6, 11), np.linspace(0.61, 0.99,39), np.linspace(0.991, 1, 10)))
    vars = {'L': L,
			'ts': ts,
			'velocities': velocities,
			'params': params,
			'directory': directory,
			'systID': systID}
    os.makedirs(directory, exist_ok=True)  # without this, gets mad that directory might not fully exist yet
    vars_file = os.path.join(directory, "variables.json")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")
    # Only save the variables file if on the first system
    if systID == 0:
        vars.update({"Date": text_date})
        saveVars(vars, vars_file)
        vars.pop("Date")
    start = wallTime()
    # runDirichlet(L, ts, velocities, params, directory, systID)
    runDirichlet(**vars)
    print(wallTime() - start, flush=True)