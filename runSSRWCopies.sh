#! /usr/bin/bash
# cd ~/Documents/code/extremeDiffusionND 


#OLD Don't need
for i in {0..30..10}; do \
	python3 -W ignore runDataAndAnalysis.py "$1" $i $2 10000 10 $3 $4 &
done


