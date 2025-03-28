#!/bin/bash
#SBATCH --account=def-nassersa-ab
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=wildfire_classification
#SBATCH --output=wildfire_classification.out
#SBATCH --error=wildfire_classification.err
#SBATCH --open-mode=truncate
#SBATCH --mail-user=iazhar@uwaterloo.ca
#SBATCH --mail-type=ALL

module load StdEnv/2020 gcc/9.3.0

# Use the full path to the Python interpreter in your conda environment:
PYENV="/home/iazhar/miniconda3/envs/wildfire_env/bin/python"

# Check that joblib and GPU-enabled xgboost are available
$PYENV -c "import joblib; print('[DEBUG] joblib version:', joblib.__version__)" || { echo "[FATAL] joblib not found in $PYENV. Exiting."; exit 1; }
$PYENV -c "import xgboost; print('[DEBUG] xgboost version:', xgboost.__version__)" || { echo "[FATAL] xgboost not found in $PYENV. Exiting."; exit 1; }

cd ~/scratch/Wildfire_risk_prediction
echo "Starting job on $(date) at $(hostname)"
echo "Using Python: $PYENV"
$PYENV scripts/modeling/main_ml_classification.py
echo "Finished job on $(date)"
