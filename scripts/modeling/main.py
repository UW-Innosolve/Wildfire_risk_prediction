import logging
import pandas as pd
import os
from model_classes.fb_regression import (
    LinearRegressionModel,
    PolynomialRegressionModel,
    LogisticRegressionModel,
    RidgeRegressionModel
)
from model_classes.fb_knn import KNNModel
from model_classes.fb_randfor import RandomForestModel
from model_classes.fb_xgboost import XGBoostModel
from model_classes.model_utils import cross_validate_model
from model_evaluation.model_reporter import Reporter
from model_classes.voting_classifier import VotingClassifierCustom, filter_models_by_threshold
import joblib  

# Configure logging: INFO level logs progress; DEBUG could be used for more details.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    logging.info("Starting Wildfire Prediction Model Training Pipeline")
    
    # ---------------------------
    # Data Loading Phase
    # ---------------------------
    
    data_dir = "scripts/modeling/model_data_dir"  # Directory where preprocessed data CSVs are stored.
    logging.info(f"Loading preprocessed data from directory: {data_dir}")
    
    # Load individual CSV files: X_train.csv, X_test.csv, y_train.csv, y_test.csv.
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
    logging.info(f"Data loaded: X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    # ---------------------------
    # (FOR LATER) Additional Feature Engineering And preprocessing
    # ---------------------------
    # For the preprocessing and feature engineering classes we made, insert them here later.
    # 
    
    # ---------------------------
    # Modeling Phase with Cross-Validation for Individual Models
    # ---------------------------
    reporter = Reporter()  # Initialize Reporter to collect evaluation metrics.
    
    # Define models to evaluate. Here we include both regression and classification models.
    # Note: Some regression models may not be appropriate for a voting classifier.
    models = {
        "KNN": KNNModel(n_neighbors=5),
        "Random Forest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
        "Logistic Regression": LogisticRegressionModel(),
        # Regression models (e.g., Linear, Polynomial, Ridge) are included for completeness,
        # but typically for classification tasks you would focus on classifiers.
        "Linear Regression": LinearRegressionModel(),
        "Polynomial Regression 3": PolynomialRegressionModel(params={'degree': 3}),
        "Polynomial Regression 5": PolynomialRegressionModel(params={'degree': 5}),
        "Polynomial Regression 7": PolynomialRegressionModel(params={'degree': 7}),
        "Ridge Regression": RidgeRegressionModel()
    }
    
    # Perform 5-fold cross-validation for each model.
    for model_name, model_obj in models.items():
        logging.info(f"Performing 5-fold cross-validation for {model_name}...")
        cv_metrics = cross_validate_model(model_obj, X_train, y_train, n_splits=5)
        logging.info(f"{model_name} (CV) average metrics: {cv_metrics}")
        reporter.add_result(model_name + " (CV)", cv_metrics)
    
    # Train on the full training set and evaluate on the hold-out test set.
    for model_name, model_obj in models.items():
        logging.info(f"Training and evaluating {model_name} on test data...")
        model_obj.train(X_train, y_train)
        test_metrics = model_obj.evaluate(X_test, y_test)
        logging.info(f"{model_name} (Test) metrics: {test_metrics}")
        reporter.add_result(model_name + " (Test)", test_metrics)
    
    # ---------------------------
    # Voting Classifier with Model Filtering
    # ---------------------------
    # For the ensemble, we typically use only the classifiers. We'll assume that
    # Logistic Regression, KNN, Random Forest, and XGBoost are the classifiers.
    # Set a threshold (e.g., minimum F1 score of 0.70) for inclusion in the ensemble.
    f1_threshold = 0.70
    logging.info(f"Filtering models with F1 score >= {f1_threshold} from test metrics for the ensemble...")
    selected_model_names = filter_models_by_threshold(reporter, threshold=f1_threshold)
    logging.info(f"Models selected for voting ensemble: {selected_model_names}")
    
    # Build a dictionary of selected models (only include classifiers).
    # Here, we filter out any regression models.
    classifier_names = {"Logistic Regression", "KNN", "Random Forest", "XGBoost"}
    selected_models = {name: models[name] for name in selected_model_names if name in classifier_names}
    
    if selected_models:
        logging.info("Training Voting Classifier (soft voting) with selected models...")
        voting_clf = VotingClassifierCustom(selected_models, voting='soft')
        voting_clf.fit(X_train, y_train)
        
        # Evaluate the ensemble on the test set.
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        voting_predictions = voting_clf.predict(X_test)
        voting_metrics = {
            "accuracy": accuracy_score(y_test, voting_predictions),
            "precision": precision_score(y_test, voting_predictions),
            "recall": recall_score(y_test, voting_predictions),
            "f1_score": f1_score(y_test, voting_predictions),
            "roc_auc": roc_auc_score(y_test, voting_clf.predict_proba(X_test))
        }
        logging.info(f"Voting Classifier (Test) metrics: {voting_metrics}")
        reporter.add_result("Voting Classifier (Test)", voting_metrics)
        
        # Save the final voting classifier model for later use (e.g., stacking)
        model_save_path = "voting_model.pkl"
        joblib.dump(voting_clf, model_save_path)
        logging.info(f"Voting Classifier model saved to {model_save_path}")
    else:
        logging.info("No models met the threshold for inclusion in the Voting Classifier.")
    
    # ---------------------------
    # Reporting Phase
    # ---------------------------
    logging.info("Generating final performance report...")
    final_report = reporter.generate_report("model_report.csv")
    logging.info("Final Model Performance Report:")
    logging.info("\n" + final_report.to_string())

if __name__ == "__main__":
    main()
