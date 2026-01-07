from dynamicRangeEvolve2DLattice import saveVars, runSystem
import numpy as np
import sys
import os
from datetime import date

if __name__ == "__main__":
    # specify these in the bash script
    L = int(sys.argv[1])
    tMax = int(sys.argv[2])
    topDirectory = sys.argv[3]
    sysID = int(sys.argv[4])
    saveInterval = int(sys.argv[5])  # in hours
    # velocities we're interested in are
    velocities = np.concatenate(
        (np.linspace(0.1, 0.6, 11),
         np.linspace(0.61, 0.99,39),
         np.linspace(0.991, 1, 10)))

    vars = {'L': L,
            'velocities': velocities,
            'tMax': tMax,
            'topDir': topDirectory,
            'sysID': sysID
            'saveInterval': saveInterval}
    print(f"vars: {vars}")
    os.makedirs(topDirectory, exist_ok=True)  # without this, gets mad that directory might not fully exist yet
    vars_file = os.path.join(topDirectory, "variables.json")
    print(f"vars_file is {vars_file}")
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    # Only save the variables file if on the first system
    if sysID == 0:
        print(f"systID is {sysID}")
        vars.update({"Date": text_date})
        print(f"vars: {vars}")
        saveVars(vars, vars_file)
        vars.pop("Date")

    runSystem(**vars)
