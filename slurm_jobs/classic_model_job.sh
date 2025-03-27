#!/bin/bash
#SBATCH --account=def-nassersa-ab   # Replace with your project account
#SBATCH --time=01:00:00             # Increase time so your job can run longer
#SBATCH --mem=32G                   # memory per node
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wildfire_classification
#SBATCH --output=wildfire_classification.out   # Always writes to the same .out
#SBATCH --error=wildfire_classification.err    # Always writes to the same .err
#SBATCH --open-mode=truncate                  # Overwrites logs each time
#SBATCH --mail-user=iazhar@uwaterloo.ca
#SBATCH --mail-type=ALL

# 1) Load modules
module load StdEnv/2020 gcc/9.3.0 python/3.8

# 2) Source conda script (to allow 'conda activate' in non-interactive shells)
source ~/miniconda3/etc/profile.d/conda.sh

# 3) Activate your conda environment
conda activate wildfire_env

# 4) Debug lines: show which python is active and joblib version
echo "==============================="
echo "[DEBUG] Using Python at: $(which python)"
python -c "import sys; print('[DEBUG] sys.executable:', sys.executable)"
python -c "import joblib; print('[DEBUG] joblib version:', joblib.__version__)"
echo "[DEBUG] CONDA_PREFIX: $CONDA_PREFIX"
echo "==============================="

# 5) Navigate to your code directory
cd ~/scratch/Wildfire_risk_prediction
echo "[DEBUG] Directory contents of $(pwd):"
ls -lh

echo "Starting job on $(date)"
echo "Running on $(hostname)"

# 6) Run your Python script
python scripts/modeling/main_ml_classification.py

echo "Finished job on $(date)"
