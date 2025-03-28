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
# Possibly: module load cuda/11.0

PYENV="/home/iazhar/miniconda3/envs/wildfire_env/bin/python"

# Check for GPU-enabled libraries:
$PYENV -c "import xgboost; print('XGBoost version:', xgboost.__version__)" || exit 1
$PYENV -c "import lightgbm; print('LightGBM version:', lightgbm.__version__)" || exit 1
$PYENV -c "import catboost; print('CatBoost installed')" || exit 1

cd ~/scratch/Wildfire_risk_prediction
echo "Starting job on $(date), on $(hostname)"


$PYENV scripts/modeling/main_xg_advanced.py
echo "Finished job on $(date)"
