#!/bin/bash

#SBATCH --time=01:00:00         #walltime
#SBATCH --ntasks=1              #number of processor cores
#SBATCH --mem-per-cpu=1G        #memory per CPU
#SBATCH -J "submit_shear"           #name the job

spacing=3.0
optimization='shearExp'

#grid
nGroups=1
withYaw=0
layout='grid'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#just_layout
nGroups=1
withYaw=0
layout='just_layout'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#one height group
nGroups=1
withYaw=0
layout='height_groups'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization





#two height groups
nGroups=2
withYaw=0
layout='height_groups'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization





#5 height groups
nGroups=5
withYaw=0
layout='height_groups'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#25 height groups
nGroups=25
withYaw=0
layout='height_groups'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#two height groups with yaw
nGroups=2
withYaw=1
layout='height_groups'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#5 height groups with yaw
nGroups=5
withYaw=1
layout='height_groups'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization




#25 height groups with yaw
nGroups=25
withYaw=1
layout='height_groups'

shearExp=0.08
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.09
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.10
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.11
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.12
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.13
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.14
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.15
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.16
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.17
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.18
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.19
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.20
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.21
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.22
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.23
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.24
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.25
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.26
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.27
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.28
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.29
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
shearExp=0.30
sbatch submitRandom.sh $spacing $shearExp $nGroups $withYaw ${SLURM_ARRAY_JOB_ID} ${SLURM_ARRAY_TASK_ID} $layout $optimization
