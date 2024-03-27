#!/bin/bash
#SBATCH --job-name=2DSphere
#SBATCH --time=0-12:00:00
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-500
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/MultiJumpRW/%A-%a.out
#SBATCH --account=jamming
#SBATCH --partition=preempt
#SBATCH --requeue

TMAX=10000
L=2000
TOPDIR=/home/jhass2/jamming/JacobData/SphericalBoundries/

mkdir -p $TOPDIR 

# tMax, L, topDir, sysID = sys.argv[1:]
python3 measureSphere.py $TMAX $L $TOPDIR $SLURM_ARRAY_TASK_ID
