import logging
import pandas as pd
import os
from model_classes.fb_regression import LinearRegressionModel, PolynomialRegressionModel, LogisticRegressionModel, RidgeRegressionModel
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
    
    # # ---------------------------
    # # Load preprocessed data 
    # # ---------------------------
    
    data_dir = "scripts/modeling/model_data_dir"
    model_data = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            if file == 'X_train.csv':
                X_train = pd.read_csv(os.path.join(data_dir, file))
            elif file == 'X_test.csv':
                X_test = pd.read_csv(os.path.join(data_dir, file))
            elif file == 'y_train.csv':
                y_train = pd.read_csv(os.path.join(data_dir, file))
            elif file == 'y_test.csv':
                y_test = pd.read_csv(os.path.join(data_dir, file))
                
    model_data = [X_train, X_test, y_train, y_test]

    # ---------------------------
    # Modeling Phase with Cross-Validation
    # ---------------------------
    reporter = Reporter()  # Initialize Reporter to collect metrics.
    
    # Define models to evaluate; parameters chosen based on preliminary experiments.
    models = {
        "KNN": KNNModel(n_neighbors=5),  # k=5 is our starting point; can tune later.
        "Random Forest": RandomForestModel(),  # Uses 100 trees and balanced class weights.
        "XGBoost": XGBoostModel(),  # Tuned for imbalanced data (scale_pos_weight=5, etc.).
        "Linear Regression": LinearRegressionModel(),
        "Polynomial Regression 3": PolynomialRegressionModel(params={'degree': 3}),
        "Polynomial Regression 5": PolynomialRegressionModel(params={'degree': 5}),
        "Polynomial Regression 7": PolynomialRegressionModel(params={'degree': 7}),
        "Logistic Regression": LogisticRegressionModel(),
        "Ridge Regression": RidgeRegressionModel()
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