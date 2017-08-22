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
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization



#just_layout
nGroups=1
withYaw=0
layout='just_layout'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization



#1 height group
nGroups=1
withYaw=0
layout='height_groups'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#2 height groups
nGroups=2
withYaw=0
layout='height_groups'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#5 height groups
nGroups=5
withYaw=0
layout='height_groups'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization





#25 height groups
nGroups=25
withYaw=0
layout='height_groups'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#2 height groups with Yaw
nGroups=2
withYaw=1
layout='height_groups'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#5 height groups with yaw
nGroups=5
withYaw=0
layout='height_groups'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization





#25 height groups with yaw
nGroups=25
withYaw=1
layout='height_groups'

spacing=2.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=2.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=3.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=4.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=5.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=6.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=7.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=8.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=9.5
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
spacing=10.0
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
