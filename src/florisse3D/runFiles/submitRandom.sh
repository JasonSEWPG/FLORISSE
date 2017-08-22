#!/bin/bash

#SBATCH --time=72:00:00         # walltime
#SBATCH --ntasks=1              # number of proc$
#SBATCH --mem-per-cpu=1G        # memory per CPU
#SBATCH --nodes=1               # number of nodes
#SBATCH -J "optimization"


module purge # removes all modules in use (such $
module load python/2/7 # loading the Python vers$
# use "module load" to load any other modules yo$

# Passing arguments into the bash script from th$
args=("$@")
spacing=${args[0]}
shearExp=${args[1]}
nGroups=${args[2]}
withYaw=${args[3]}
jobID=${args[4]}
taskID=${args[5]}
layout=${args[6]}
optimization=${args[7]}

python optRandom.py $spacing $shearExp $nGroups $withYaw $jobID $taskID $layout $optimization

exit 0
