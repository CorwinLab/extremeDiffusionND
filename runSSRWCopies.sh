#! /usr/bin/bash
# cd ~/Documents/code/extremeDiffusionND 

for i in {0..30..10}; do \
	python3 -W ignore runDataAndAnalysis.py "$1" $i 1e10 100000 10 $2 &
done


