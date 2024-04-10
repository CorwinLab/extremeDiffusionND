#! /usr/bin/bash
# cd ~/Documents/code/extremeDiffusionND 
#    directoryName = sys.argv[1] # String
#    systID = int(sys.argv[2]) # Integer
#    occupancy = int(float(sys.argv[3])) # Integer, cast from float to allow for scientific notation
#    MaxT = int(sys.argv[4]) # Integer
#    #dirichlet = eval(sys.argv[5]) # Bool
#    distribution = str(sys.argv[5]) #string for distribution name
#    params = sys.argv[6] #i think this needs to be a list
#    PDF = eval(sys.argv[7]) # Bool
#    numSystems = int(sys.argv[8]) # Integer
for i in {0..90..10}; do \
	python3 -W ignore runDataAndAnalysis.py "$1" $i $2 10000 $3 $4 $5 10 &
done

