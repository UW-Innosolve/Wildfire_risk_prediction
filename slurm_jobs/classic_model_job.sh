#!/bin/bash
#SBATCH --account=def-nassersa-ab      # Replace with your actual account
#SBATCH --time=00:06:00               # hh:mm:ss or d-hh:mm:ss
#SBATCH --mem=32G                     # memory per node
#SBATCH --cpus-per-task=8             # number of CPU cores
#SBATCH --job-name=wildfire_classification
#SBATCH --output=%x-%j.out            # wildfire_classification-<jobID>.out
#SBATCH --error=%x-%j.err             # wildfire_classification-<jobID>.err
#SBATCH --mail-user=iazhar@uwaterloo.ca
#SBATCH --mail-type=ALL               # or BEGIN,END,FAIL

# Load modules
module load StdEnv/2020 gcc/9.3.0 python/3.8

# (1) Source conda script to enable 'conda activate' in non-interactive shells
source ~/miniconda3/etc/profile.d/conda.sh

# (2) Activate your environment
conda activate wildfire_env

# (3) Print debug info about which Python is active
echo "==============================="
echo "[DEBUG] Which python: $(which python)"
python -c "import sys; print('[DEBUG] sys.executable:', sys.executable)"
python -c "import joblib; print('[DEBUG] joblib version:', joblib.__version__)"
echo "[DEBUG] CONDA_PREFIX: $CONDA_PREFIX"
echo "==============================="

# (4) Navigate to your project directory
cd ~/scratch/Wildfire_risk_prediction
echo "[DEBUG] Directory contents in $(pwd):"
ls -lh

echo "Starting job on $(date)"
echo "Running on $(hostname)"

# (5) Run your Python script
python scripts/modeling/main_ml_classification.py

echo "Finished job on $(date)"
