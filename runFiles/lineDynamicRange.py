from dynamicRangeEvolve2DLattice import saveVars, runSystemLine
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
    saveInterval = float(sys.argv[5])  # in hours
    # saveInterval = float(1/60)  # 1 minute for testing
    print(f"sysID: {sysID}", flush=True)
    # velocities we're interested in are
    velocities = np.geomspace(1e-2, 3)  # 50 of these???

    variables = {'L': L,
            'velocities': velocities,
            'tMax': tMax,
            'topDir': topDirectory,
            'sysID': sysID,
            'saveInterval': saveInterval}
    # .../$TMAX/LINE/0.npy for past a line or .../L$L/LINE/Final0.npy
    newTopDir = os.path.join(topDirectory, "Line")  # /projects/jamming/fransces/data/.../L$L/Line/    os.makedirs(newTopDir, exist_ok=True)
    os.makedirs(newTopDir, exist_ok=True)  # without this, gets mad that directory might not fully exist yet
    vars_file = os.path.join(newTopDir, "variables.json")
    print(f"vars_file is {vars_file}")
    print(f"vars: {variables}", flush=True)
    today = date.today()
    text_date = today.strftime("%b-%d-%Y")

    # Only save the variables file if on the first system
    if sysID == 0:
        print(f"systID is {sysID}")
        variables.update({"Date": text_date})
        print(f"vars: {variables}", flush=True)
        saveVars(variables, vars_file)
        variables.pop("Date")

    runSystemLine(**variables)
