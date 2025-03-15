#!/bin/bash
#SBATCH --account=jamming
#SBATCH --job-name=EigenValues
#SBATCH --output=/home/jhass2/jamming/JacobData/logs/EigenValues/%A.%a.out
#SBATCH --error=/home/jhass2/jamming/JacobData/logs/EigenValues/%A.%a.err
#SBATCH --partition=preempt
#SBATCH --ntasks=1
#SBATCH --array=500-9999%1000
#SBATCH --time=0-06:00:00
#SBATCH --requeue

DIRECTORYNAME="/projects/jamming/shared/LargestEigenValue/$SLURM_ARRAY_TASK_ID"
mkdir -p $DIRECTORYNAME
matlab -nodisplay -nodesktop -nosplash -batch "largestEigenValue('$DIRECTORYNAME', '$SLURM_ARRAY_TASK_ID'); exit"
