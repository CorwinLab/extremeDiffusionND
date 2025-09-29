import numpy as np
import npquad
import os
from time import time as wallTime  # start = wallTime() to avoid issues with using time as variable
from datetime import date
import sys
import memEfficientEvolve2DLattice as m

if __name__ == "__main__":
    # Test Code
    # L, tMax, distName, params, directory, systID = 5000, 10000, 'Dirichlet', '1,1,1,1', './', 0

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
        print(f"params: {params}")

    ts = m.getListOfTimes(tMax - 1, 1)  # 1 to 10,000-1
    velocities = np.linspace(0.1, 1, 46)  # 0.1 to 1 in increments of 0.02

    vars = {'L': L,
            'ts': ts,
            'velocities': velocities,
            'distName': distName,
            'params': params,
            'directory': directory,
            'systID': systID}
    print(f"vars: {vars}")
    os.makedirs(directory, exist_ok=True)  # without this, gets mad that directory might not fully exist yet
    vars_file = os.path.join(directory, "variables.json")
    print(f"vars_file is {vars_file}")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    # Only save the variables file if on the first system
    if systID == 0:
        print(f"systID is {systID}")
        vars.update({"Date": text_date})
        m.saveVars(vars, vars_file)
        vars.pop("Date")

    start = wallTime()
    m.runSystem(**vars)
    print(wallTime() - start, flush=True)
