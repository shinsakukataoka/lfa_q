#! /bin/bash
#SBATCH --partition=gpu-a100-q         # Use the GPU partition
#SBATCH --gres=gpu:a100:1             # Request 1 A100 GPU
#SBATCH --mem=30G                     # Request 30GB memory
#SBATCH --time=02:00:00               # Job runtime limit: 2 hours
#SBATCH --output=logs/dgpu-%j.out     # Standard output (organized into logs folder)
#SBATCH --error=logs/dgpu-%j.err      # Standard error

# Load CUDA and cuDNN modules
module load cuda11.7/toolkit
module load cudnn8.5-cuda11.7

# Activate your Python virtual environment
source env/bin/activate
cd /home/skataoka26/project/AI/Skin_RL

# Use nvidia-smi to log GPU usage periodically
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,nounits -l 10 > logs/gpu_usage-%j.csv &

# Start time tracking
start_time=$(date +%s)

# Run your Python script
srun python RL_Skin_Cancer_Demo_Management.py --n_patients 100 --n_episodes 150 --n_actions 2

# End time tracking
end_time=$(date +%s)

# Calculate total runtime
echo "Total runtime: $((end_time - start_time)) seconds" >> logs/runtime-%j.log

# Log CPU and memory usage
free -h > logs/memory_usage-%j.log
top -b -n 1 | head -n 20 >> logs/cpu_usage-%j.log

# Kill nvidia-smi monitoring
pkill -f nvidia-smi
