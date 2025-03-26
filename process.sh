#!/bin/bash
#SBATCH --account=def-nassersa-ab
#SBATCH --mail-user=teo.vujovic@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=50G
#SBATCH --time=0-01:00

cd /home/tvujovic/scratch/firebird/feat-eng/Wildfire_risk_prediction/scripts/data_processing/
source dataprocess/bin/activate

python main.py
