#!/bin/bash
#SBATCH --account=def-nassersa-ab
#SBATCH --time=72:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --job-name=wildfire_classification
#SBATCH --output=wildfire_classification.out
#SBATCH --error=wildfire_classification.err
#SBATCH --open-mode=truncate
#SBATCH --mail-user=iazhar@uwaterloo.ca
#SBATCH --mail-type=ALL

module load StdEnv/2020 gcc/9.3.0
# If required by your environment, load the appropriate CUDA module. For example:
# module load cuda/12.6

# Use the full path to the Python interpreter in your conda environment:
PYENV="/home/iazhar/miniconda3/envs/wildfire_env/bin/python"

# Check that the necessary GPU-enabled libraries are available:
$PYENV -c "import xgboost; print('XGBoost version:', xgboost.__version__)" || { echo "[FATAL] xgboost not found."; exit 1; }
$PYENV -c "import lightgbm; print('LightGBM version:', lightgbm.__version__)" || { echo "[FATAL] lightgbm not found."; exit 1; }
$PYENV -c "import catboost; print('CatBoost installed');" || { echo "[FATAL] catboost not found."; exit 1; }

cd ~/scratch/Wildfire_risk_prediction
echo "Starting job on $(date) at $(hostname)"
echo "Using Python: $PYENV"

# Quick check: is the GPU visible to the system?
echo "Running nvidia-smi to confirm GPU availability..."
nvidia-smi || echo "No GPU driver or GPU not found on this node!"

echo "Launching pipeline script..."
$PYENV scripts/modeling/main_deep_learning.py

echo "Finished job on $(date)"
