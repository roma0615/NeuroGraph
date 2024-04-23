#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01-00:00:00
#SBATCH --gres=gpu:1
#

# prepare environment
module load cudatoolkit/12.2
conda activate neurograph

./run_baseline.sh
