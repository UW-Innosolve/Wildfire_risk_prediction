import logging
import pandas as pd
from scripts.data_processing.preprocessor import Preprocessor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

data_dir = "scripts/data_processing/raw_data_dir"
logging.info(f"Pulling raw data from directory: {data_dir}")

###########################

# Initialize the Preprocessor.
preprocessor = Preprocessor(data_dir)
logging.info("Loading data from CSV files...")
data = preprocessor.load_data()  # Aggregate CSVs.

logging.info("Cleaning data (converting dates, removing missing target values)...")
data = preprocessor.clean_data()  # Clean the data.

logging.info("Performing feature engineering (e.g., extracting month from date)...")
data = preprocessor.feature_engineering()  # Feature engineering.

# Define the list of features based on our dataset headers.
features = [
    '2t', '2d', '10u', '10v', 'sp', 'tp', #'relative_humidity', 'atmospheric_dryness',
    'latitude', 'longitude', 'month',
    'lightning_count', 'absv_strength_sum', 'multiplicity_sum',
    'railway_count', 'power_line_count', 'highway_count', 'aeroway_count', 'waterway_count'
]
target = 'is_fire_day'
logging.info(f"Selected features: {features}")
logging.info(f"Target variable: {target}")

# Scale features; we exclude 'date' because it's not a numeric predictor.
logging.info("Scaling features using StandardScaler...")
preprocessor.scale_features(features)

# Split the data; apply SMOTE for balancing minority class (fire days).
logging.info("Splitting data into training and test sets and applying SMOTE for balancing...")
X_train, X_test, y_train, y_test = preprocessor.split_data(features, target, apply_smote=True)
logging.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    