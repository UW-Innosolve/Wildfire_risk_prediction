import logging
from data_preprocessing.preprocessor import Preprocessor
from model_classes.fb_log_regression import LogisticRegressionModel
from model_classes.fb_knn import KNNModel
from model_classes.fb_randfor import RandomForestModel
from model_classes.fb_xgboost import XGBoostModel
from model_classes.model_utils import cross_validate_model
from model_evaluation.model_reporter import Reporter

# Configure logging: INFO level logs progress, DEBUG could be used for more details.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    logging.info("Starting Wildfire Prediction Model Training Pipeline")
    
    # ---------------------------
    # Preprocessing Phase
    # ---------------------------
    data_dir = "/Users/ibzazh/Documents/test_data"

    #data_dir = r"C:\Users\ibuaz\OneDrive\Desktop\firebird_data\output_data"  # Update this path as needed.
    logging.info(f"Using data directory: {data_dir}")
    
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
    
    # ---------------------------
    # Modeling Phase with Cross-Validation
    # ---------------------------
    reporter = Reporter()  # Initialize Reporter to collect metrics.
    
    # Define models to evaluate; parameters chosen based on preliminary experiments.
    models = {
        "Logistic Regression": LogisticRegressionModel(),
        "KNN": KNNModel(n_neighbors=5),  # k=5 is our starting point; can tune later.
        "Random Forest": RandomForestModel(),  # Uses 100 trees and balanced class weights.
        "XGBoost": XGBoostModel()  # Tuned for imbalanced data (scale_pos_weight=5, etc.).
    }
    
    # Perform K-Fold cross-validation (5 folds) for each model.
    for model_name, model_obj in models.items():
        logging.info(f"Performing 5-fold cross-validation for {model_name}...")
        cv_metrics = cross_validate_model(model_obj, X_train, y_train, n_splits=5)
        logging.info(f"Average CV metrics for {model_name}: {cv_metrics}")
        reporter.add_result(model_name + " (CV)", cv_metrics)
    
    # Train on the full training set and evaluate on the hold-out test set.
    for model_name, model_obj in models.items():
        logging.info(f"Training and evaluating {model_name} on test data...")
        model_obj.train(X_train, y_train)
        test_metrics = model_obj.evaluate(X_test, y_test)
        logging.info(f"{model_name} Test Metrics: {test_metrics}")
        reporter.add_result(model_name + " (Test)", test_metrics)
    
    # ---------------------------
    # Reporting Phase
    # ---------------------------
    logging.info("Generating final performance report...")
    final_report = reporter.generate_report("model_report.csv")
    logging.info("Final Model Performance Report:")
    logging.info(f"\n{final_report}")

if __name__ == "__main__":
    main()


## NOTE:
## Final model should use a sliding window approach, possibly expanding window approach
## Possibly train a seperate model for each timeframe of prediction/forecast (ie. how many days in the future)
## Possibly train a model that predicts the full set of 5 days (ie. a list with days 1-5)