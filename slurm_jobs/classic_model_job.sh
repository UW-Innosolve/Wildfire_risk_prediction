#!/bin/bash
#SBATCH --account=<your-account>       # e.g. def-somepi
#SBATCH --time=08:00:00               # hh:mm:ss or d-hh:mm:ss
#SBATCH --mem=32G                     # memory per node
#SBATCH --cpus-per-task=8             # number of CPU cores
#SBATCH --job-name=wildfire_classification
#SBATCH --output=%x-%j.out            # console output goes to: wildfire_classification-<jobID>.out
#SBATCH --error=%x-%j.err             # console errors go to: wildfire_classification-<jobID>.err

# Load modules/activate environment
module load StdEnv/2020  gcc/9.3.0  python/3.8
conda activate wildfire_env

# Navigate to the code directory (adjust the path as needed)
cd ~/scratch/Wildfire_risk_prediction

# Optional: log what we're doing
echo "Starting job on $(date)"
echo "Running on $(hostname)"

# Run the script (the one you pasted in your question)
python scripts/modeling/main_ml_classification.py

echo "Finished job on $(date)"
