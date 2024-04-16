#!/bin/bash
#SBATCH --job-name=2DSphere
#SBATCH --time=1-00:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/t1Regime/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-4999
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/t1Regime/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=5000
L=2000
DISTRIBUTION='uniform'
PARAMS='None'
TOPDIR=/home/jhass2/jamming/JacobData/2DHistograms/$DISTRIBUTION

mkdir -p $TOPDIR

for i in {0..9}
do
    # tMax, L, topDir, distribution, params, sysID = sys.argv[1:]
    SYSID=$(($SLURM_ARRAY_TASK_ID * 10 + i))
    python3 measureLineProb.py $TMAX $L $TOPDIR $DISTRIBUTION $PARAMS $SYSID
done
