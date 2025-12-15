#!/bin/bash

TMAX=1000
DISTRIBUTION="Dirichlet"
ALPHA=1
PARAMS=$ALPHA,$ALPHA,$ALPHA,$ALPHA
L=1000
DIRECTORYNAME="/home/fransces/Documents/code/extremeDiffusionND/tests/debugOccload/"
ARRAYID=4

python3 runFiles/runDirichlet.py $L $TMAX $DISTRIBUTION $PARAMS $DIRECTORYNAME $ARRAYID

