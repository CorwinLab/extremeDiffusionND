#!/bin/bash
#SBATCH --account=jamming
#SBATCH --job-name=measureDelta
#SBATCH --output=logs/measureAlpha0.03162278/%A.%a.out
#SBATCH --error=logs/measureAlpha0.03162278/%A.%a.err
#SBATCH --partition=computelong,memorylong
#SBATCH --ntasks=1
#SBATCH --array=0-499
#SBATCH --time=10-00:00:00
#SBATCH --mem-per-cpu=40GB
#SBATCH --requeue

TMAX=10000
PARAMS='None'
DISTRIBUTION='delta'
WIDTH=1
L=5000
DIRECTORYNAME="/projects/jamming/fransces/data/memoryEfficientMeasurements/dirichlet/ALPHA$ALPHA/L$L/tMax$TMAX"
ARRAYID=$SLURM_ARRAY_TASK_ID

python3 extremeDiffusionND/memEfficientEvolve2DLattice.py $L $TMAX $DISTRIBUTION $PARAMS $WIDTH $DIRECTORYNAME $ARRAYID