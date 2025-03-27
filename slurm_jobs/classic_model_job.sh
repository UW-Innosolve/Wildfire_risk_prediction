#!/bin/bash
#SBATCH --account=<iazhar>       # e.g. def-somepi
#SBATCH --time=00:06:00               # hh:mm:ss or d-hh:mm:ss
#SBATCH --mem=32G                     # memory per node
#SBATCH --cpus-per-task=8             # number of CPU cores
#SBATCH --job-name=wildfire_classification
#SBATCH --output=%x-%j.out            # console output goes to: wildfire_classification-<jobID>.out
#SBATCH --error=%x-%j.err             # console errors go to: wildfire_classification-<jobID>.err
#SBATCH --mail-user=iazhar@uwaterloo.ca     # your email address
#SBATCH --mail-type=ALL                     # or BEGIN,END,FAIL

# Load modules/activate your environment
module load StdEnv/2020 gcc/9.3.0 python/3.8
conda activate wildfire_env

# Navigate to the directory containing your code
cd ~/scratch/Wildfire_risk_prediction

echo "Starting job on $(date)"
echo "Running on $(hostname)"

# Run your Python script
python scripts/modeling/main_ml_classification.py

echo "Finished job on $(date)"
