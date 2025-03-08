#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=200G             
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=8    
#SBATCH --output=run_hmm.out

echo "Starting job at $(date)"
echo "Running on node $(hostname)"

module load neuroimaging-env
module load gcc/12.3.0 cuda/12.2.1

echo "Modules loaded. Launching Python script..."

srun python -u /m/nbe/scratch/hmmpipeline/run_hmm.py

echo "Job finished at $(date)"