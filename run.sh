#! /bin/bash
#SBATCH --partition=gpu-a100-q         # Use the GPU partition
#SBATCH --gres=gpu:a100:1             # Request 1 A100 GPU
#SBATCH --mem=30G                     # Request 30GB memory
#SBATCH --time=02:00:00               # Job runtime limit: 30 minutes
#SBATCH --output=dgpu-%j.out          # Standard output
#SBATCH --error=dgpu-%j.err           # Standard error

module load cuda11.7/toolkit
module load cudnn8.5-cuda11.7

source env/bin/activate
cd :/home/skataoka26/project/AI/Skin_RL
# srun python RL_Skin_Cancer_Demo_Diagnosis.py --n_patients 100 --n_episodes 150 --use_unknown False <- in case of simple diagnosis (instead of SL)
# run python RL_Skin_Cancer_Original.py
# srun python Skin_Cancer_RL_Demo_Patient_Management.py --n_patients 1 --n_episodes 130 --n_actions 3 <- in case of patient_management

# this should be the baseline - the implementation specified in the paper
# srun python RL_Skin_Cancer_Demo_Management.py --n_patients 100 --n_episodes 150 --n_actions 2
# our q learnig method
# srun python RL_Skin_Cancer_Demo_Management_QL.py --n_patients 100 --n_episodes 150 --n_actions 2
# our demo DQN
# srun python RL_Skin_Cancer_Demo_DoubleDQN.py --n_patients 100 --n_episodes 150 --n_actions 2
# our q learning with func approx.
# srun python RL_Skin_Cancer_Demo_Management_QL_Linear.py --n_patients 100 --n_episodes 150 --n_actions 2
# try looping:
# srun python RL_Skin_Cancer_QL_Tabular_Loop.py --n_patients 100 --n_episodes 150 --n_actions 2
srun python RL_Skin_Cancer_Demo_Management_QL_Linear_Learning_Decay.py --n_patients 100 --n_episodes 150 --n_actions 2

