#! /bin/bash
#SBATCH --partition=gpu-a100-q         # Use the GPU partition
#SBATCH --gres=gpu:a100:1             # Request 1 A100 GPU
#SBATCH --mem=30G                     # Request 30GB memory
#SBATCH --time=02:00:00               # Job runtime limit: 30 minutes
#SBATCH --output=dgpu-%j.out          # Standard output
#SBATCH --error=dgpu-%j.err           # Standard error

# Load CUDA and cuDNN modules
module load cuda11.7/toolkit
module load cudnn8.5-cuda11.7

# Activate your Python virtual environment
source env/bin/activate
cd :/home/skataoka26/project/AI/Skin_RL
# Run your Python script
srun python RL_Skin_Cancer_Demo_Diagnosis.py --n_patients 100 --n_episodes 150 --use_unknown False
