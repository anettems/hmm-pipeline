#!/bin/bash
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --time=05:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --output=run_hmm-beta.out

echo "Starting job at $(date)"
echo "Running on node $(hostname)"

module load neuroimaging-env
module load gcc/12.3.0 cuda/12.2.1

echo "Modules loaded. Launching Python script..."

srun python -u /m/nbe/scratch/hmmpipeline/run_hmm.py

echo "Job finished at $(date)"