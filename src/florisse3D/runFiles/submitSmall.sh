#!/bin/bash

#SBATCH --time=01:00:00         #walltime
#SBATCH --ntasks=1              #number of processor cores
#SBATCH --mem-per-cpu=1G        #memory per CPU
#SBATCH -J "submit_spacing"           #name the job

shearExp=0.1
optimization='spacing'

#grid
nGroups=1
withYaw=0
layout='grid'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
