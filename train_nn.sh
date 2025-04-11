#!/bin/bash
#SBATCH --account=def-nassersa-ab
#SBATCH --mail-user=teo.vujovic@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-06:00

cd /home/tvujovic/scratch/
source dataprocess/bin/activate
cd ./firebird/Wildfire_risk_prediction/scripts/modeling/

python main.py
