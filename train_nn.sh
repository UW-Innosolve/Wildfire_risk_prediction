#!/bin/bash
#SBATCH --account=def-nassersa-ab
#SBATCH --mail-user=teo.vujovic@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=100G
#SBATCH --time=0-01:00

cd /home/tvujovic/scratch/
source dataprocess/bin/activate
cd ./firebird/Wildfire_risk_prediction/scripts/modeling/

python main.py
