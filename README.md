# Wildfire Risk Prediction Project

Overview of project structure:
wildfire-risk-prediction/
│
├── data/                   
│   ├── fire_days/                # Data for fire occurrence days ("ones")
│   ├── non_fire_days/            # Data for non-fire days ("zeroes")
│   ├── external/                 # Copernicus ERA5 data and other external sources
│   ├── processed/                # Preprocessed datasets ready for modeling
│   ├── interim/                  # Temporary storage for intermediate datasets
│   └── raw/                      # Original unprocessed data files
│
├── notebooks/                    # Jupyter notebooks for exploration and initial analysis
│   ├── 01-fire-data-exploration.ipynb
│   ├── 02-era5-data-exploration.ipynb
│   ├── 03-data-preprocessing.ipynb
│   ├── 04-feature-engineering.ipynb
│   └── 05-model-training.ipynb
│
├── scripts/                      # Python scripts for data processing and modeling
│   ├── data_collection/          # Scripts for data collection
│   ├── data_processing/          # Scripts for data cleaning and merging
│   ├── feature_engineering/      # Scripts for feature creation
│   ├── modeling/                 # Scripts for model training and evaluation
│   └── utils/                    # Helper functions for logging, configuration, etc.
│
├── slurm_jobs/                   # SLURM job scripts for running on HPC clusters
│   ├── collect_data_job.sbatch
│   ├── preprocess_data_job.sbatch
│   ├── train_model_job.sbatch
│   └── evaluate_model_job.sbatch
│
├── environments/                 # Environment configuration files
│   ├── requirements.txt          # Python package requirements
│   ├── environment.yml           # Conda environment file
│   └── setup.sh                  # Script to set up the environment
│
├── reports/                      # Generated reports, figures, and logs
│   ├── figures/                  # Visualizations and plots
│   └── logs/                     # Log files for tracking experiments and job outputs
│
├── tests/                        # Unit and integration tests
│   ├── test_data_processing.py
│   ├── test_feature_engineering.py
│   └── test_modeling.py
│
├── .gitignore                    # List of files and directories to ignore in version control
├── README.md                     # Overview and setup instructions for the project
└── LICENSE                       # License file (if applicable)





Clone the Repository:

''
git clone https://github.com/your_username/wildfire-risk-prediction.git
cd wildfire-risk-prediction
''

Set Up the Environment:

If using a virtual environment:
'''
python3 -m venv environments/venv
source environments/venv/bin/activate
pip install -r environments/requirements.txt
'''

Runing Jobs on cedar / SLURM job management:

To run tasks on the Cedar cluster, use the SLURM job scripts provided in the slurm_jobs/ directory. 
Submit jobs using the sbatch command:

# Submit a job for data collection
''sbatch slurm_jobs/collect_data_job.sbatch'''

# Submit a job for model training
''sbatch slurm_jobs/train_model_job.sbatch''

You can monitor job status with:
''
squeue -u your_username
''