#! /usr/bin/bash
# cd ~/Documents/code/extremeDiffusionND 
#    directoryName = sys.argv[1] # String
#    systID = int(sys.argv[2]) # Integer
#    occupancy = int(float(sys.argv[3])) # Integer, cast from float to allow for scientific notation
#    MaxT = int(sys.argv[4]) # Integer
#    #dirichlet = eval(sys.argv[5]) # Bool
#    distribution = str(sys.argv[5]) #string for distribution name
#    params = sys.argv[6] #i think this needs to be a list
#    PDF = eval(sys.argv[7]) # Bool (keep this one b/c I need it in runDataAndAnalysis)
#    numSystems = int(sys.argv[8]) # Integer
#    absorbingradius = sys.argv[9] # 'off' or None or a number

# ex call: bash runCopiesOfDataAndAnalysis.sh "path" 1e10 100 uniform None False 80
# runCopies = $0, "path" = $1, 1e10 = $2, 100 = $3, uniform = $4, None = $5, False= $6, 80 = $7
for i in {0..10..10}; do \
  # path, occ, systID, #maxT, distribution, params, PDF, numSystems?
	python3 -W ignore runDataAndAnalysis.py "$1" $i $2 $3 $4 $5 $6 10 $7 &
done

