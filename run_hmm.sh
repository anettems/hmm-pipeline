#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=16G             
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=4      
#SBATCH --output=run_hmm.out

# Load necessary modules
module load neuroimaging-env
module load gcc/12.3.0 cuda/12.2.1


# Run your Python script
python3 run_hmm.py
