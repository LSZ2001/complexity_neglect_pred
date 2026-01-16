#!/bin/bash

#SBATCH -J ShuzeTest                      # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liushuze@login.rc.fas.harvard.edu   # Where to send mail
#SBATCH --ntasks=1                                  # Run a single task, defaults to single CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10gb
#SBATCH --time=10:00:00                          # Time limit hrs:min:sec
#SBATCH --array=0
#SBATCH -o ./simplicity_bias/sh_files/test."%j"_"%a".out                            # Standard output to current dir
#SBATCH -e ./simplicity_bias/sh_files/test."%j"_"%a".err                             # Error output to current dir
 

# Enable Additional Software
module load python/3.10.9-fasrc01

# Run the job commands
conda run -n pyro_env_new2 python ./simplicity_bias/simplicitybias_glme_null.py