#! /usr/bin/bash
# cd ~/Documents/code/extremeDiffusionND 
#    directoryName = sys.argv[1]
#    systID = int(sys.argv[2])
#    occupancy = int(float(sys.argv[3]))
#    MaxT=int(sys.argv[4])
#    dirichlet = sys.argv[5]
#    PDF = sys.argv[6]
#    numSystems = int(sys.argv[7])
for i in {0..10..10}; do \
	python3 -W ignore runDataAndAnalysis.py "$1" $i $2 10000 $3 $4 10 &
done

