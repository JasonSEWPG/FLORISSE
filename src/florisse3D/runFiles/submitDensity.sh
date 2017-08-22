#!/bin/bash

#SBATCH --time=01:00:00         #walltime
#SBATCH --ntasks=1              #number of processor cores
#SBATCH --mem-per-cpu=4G        #memory per CPU
#SBATCH -J "submit_density"           #name the job

shearExp=0.1
optimization='spacing'

#grid
nGroups=1
withYaw=0
layout='grid'

density=0.025
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.05
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.075
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.1
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.125
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.15
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.175
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.2
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.225
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.25
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.275
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.3
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization


#just_layout
nGroups=1
withYaw=0
layout='just_layout'

density=0.025
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.05
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.075
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.1
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.125
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.15
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.175
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.2
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.225
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.25
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.275
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.3
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization

#1 height group
nGroups=1
withYaw=0
layout='height_groups'

density=0.025
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.05
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.075
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.1
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.125
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.15
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.175
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.2
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.225
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.25
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.275
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.3
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization

#2 height groups
nGroups=2
withYaw=0
layout='height_groups'

density=0.025
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.05
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.075
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.1
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.125
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.15
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.175
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.2
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.225
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.25
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.275
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.3
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization


#2 height groups with Yaw
nGroups=2
withYaw=1
layout='height_groups'

density=0.025
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.05
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.075
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.1
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.125
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.15
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.175
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.2
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.225
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.25
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.275
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
density=0.3
sbatch submitRandom.sh $density $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
