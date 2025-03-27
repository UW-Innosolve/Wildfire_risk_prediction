#!/bin/bash
#SBATCH --account=def-nassersa-ab     # Replace with your actual account
#SBATCH --time=01:00:00               # 1 hour; adjust as needed
#SBATCH --mem=32G                     # memory per node
#SBATCH --cpus-per-task=8             # number of CPU cores
#SBATCH --job-name=wildfire_classification
#SBATCH --output=wildfire_classification.out     # Overwrites each run
#SBATCH --error=wildfire_classification.err      # Overwrites each run
#SBATCH --open-mode=truncate
#SBATCH --mail-user=iazhar@uwaterloo.ca
#SBATCH --mail-type=ALL

# 1) Load modules
module load StdEnv/2020 gcc/9.3.0 python/3.8

# 2) Source the conda script (so we can 'conda activate' in a non-interactive shell)
source ~/miniconda3/etc/profile.d/conda.sh

# 3) Activate your environment
conda activate wildfire_env

# 4) Immediately verify joblib is available in this environment
echo "==============================="
echo "[DEBUG] Using Python at: $(which python)"
python -c "import sys; print('[DEBUG] sys.executable:', sys.executable)"

python -c "import joblib; print('[DEBUG] joblib version:', joblib.__version__)" || {
    echo "[FATAL] joblib is not found in the currently active conda environment. Exiting."
    exit 1
}
echo "[DEBUG] CONDA_PREFIX is: $CONDA_PREFIX"
echo "==============================="

# 5) Navigate to code directory
cd ~/scratch/Wildfire_risk_prediction
echo "[DEBUG] Directory contents in $(pwd):"
ls -lh

echo "Starting job on $(date) at $(hostname)"

# 6) Run your Python script
python scripts/modeling/main_ml_classification.py

echo "Finished job on $(date)"
