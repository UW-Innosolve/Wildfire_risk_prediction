#!/bin/bash
#SBATCH --account=def-nassersa-ab     # Replace with your actual account
#SBATCH --time=24:00:00              # 24 hours (adjust as needed)
#SBATCH --mem=128G                   # Memory per node
#SBATCH --cpus-per-task=8            # Number of CPU cores
#SBATCH --gres=gpu:1                 # Request one GPU
#SBATCH --job-name=wildfire_classification
#SBATCH --output=wildfire_classification.out  # Overwrites each run
#SBATCH --error=wildfire_classification.err   # Overwrites each run
#SBATCH --open-mode=truncate
#SBATCH --mail-user=iazhar@uwaterloo.ca
#SBATCH --mail-type=ALL

module load StdEnv/2020 gcc/9.3.0

# Use the full path to the Python interpreter in your conda environment:
PYENV="/home/iazhar/miniconda3/envs/wildfire_env/bin/python"

# Check that joblib is available:
$PYENV -c "import joblib; print('[DEBUG] joblib version:', joblib.__version__)" || {
    echo "[FATAL] joblib not found in $PYENV. Exiting."
    exit 1
}

cd ~/scratch/Wildfire_risk_prediction
echo "Starting job on $(date) at $(hostname)"
echo "Using Python: $PYENV"
$PYENV scripts/modeling/main_ml_classification.py
echo "Finished job on $(date)"
