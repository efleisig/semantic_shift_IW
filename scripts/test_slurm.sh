#!/bin/bash
#SBATCH --job-name=slurm-test    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=80G        # memory per cpu-core (4G is default)
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=efleisig@princeton.edu

echo 'Hello world!'
module load anaconda3/5.3.1
python semantic_shift_analysis.py