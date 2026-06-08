from dynamicRangeEvolve2DLattice import saveVars, runSystemCircle
import numpy as np
import sys
import os
from datetime import date
import time

if __name__ == "__main__":
    # startup jitter to spread out the load on talapas
    time.sleep(np.random.uniform(0,30))
    # specify these in the bash script
    L = int(sys.argv[1])
    tMax = int(sys.argv[2])
    topDirectory = sys.argv[3]
    occDirectory = sys.argv[4]
    sysID = int(sys.argv[5])
    saveInterval = float(sys.argv[6])  # in hours
    alpha = float(sys.argv[7])
    # velocities we're interested in are
    velocities = np.concatenate(
        (np.linspace(0.1, 0.6, 11),
         np.linspace(0.61, 0.99,39),
         np.linspace(0.991, 1, 10)))

    variables = {'L': L,
                 'velocities': velocities,
                 'tMax': tMax,
                 'topDir': topDirectory,
                 'occDir': occDirectory,
                 'sysID': sysID,
                 'saveInterval': saveInterval,
                 'alpha': alpha}
    # os.makedirs(topDirectory, exist_ok=True)  # without this, gets mad that directory might not fully exist yet
    vars_file = os.path.join(topDirectory, "variables.json")
    print(f"vars_file is {vars_file}")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    # Only save the variables file if on the first system
    if sysID == 0:
        # print(f"systID is {sysID}")
        variables.update({"Date": text_date})
        # print(f"vars: {variables}", flush=True)
        saveVars(variables, vars_file)
        variables.pop("Date")
    print(f"vars: {variables}", flush=True)

    runSystemCircle(**variables)
