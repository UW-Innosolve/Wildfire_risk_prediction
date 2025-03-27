import logging
import os
import joblib
import pandas as pd
import numpy as np

# === Classification models ===
from model_classes.fb_knn import KNNModel
from model_classes.fb_randfor import RandomForestModel
from model_classes.fb_xgboost import XGBoostModel
from model_classes.fb_regression import LogisticRegressionModel

# === Utilities for cross-validation ===
from model_classes.model_utils import cross_validate_model

# === Reporting and voting classifier ===
from model_evaluation.model_reporter import Reporter
from model_classes.voting_classifier import VotingClassifierCustom, filter_models_by_threshold

# -----------------------------------------------------------------------------
# Configure Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------------------------------------------------------
# Sliding Window Function
# -----------------------------------------------------------------------------
def create_sliding_windows(X, y, lag_window=14, forecast_window=5, agg_func=np.mean):
    """
    Create training instances using a sliding window for a forecasting approach.
    
    For each training instance:
      - The input features are the flattened data of the previous lag_window days
      - The target is the aggregated value (using agg_func) of the *next* forecast_window days from y

    Parameters:
      X : Pandas DataFrame (indexed in time order)
      y : Pandas Series (aligned with X)
      lag_window : int, number of past days to use as input
      forecast_window : int, number of future days to aggregate into target
      agg_func : function to aggregate the future values (e.g., np.mean, np.sum)

    Returns:
      X_new : ndarray of shape (n_samples, lag_window * n_features)
      y_new : ndarray of shape (n_samples,)
    """
    n_samples = len(y) - lag_window - forecast_window + 1
    if n_samples <= 0:
        raise ValueError("Not enough data to create the specified sliding windows.")

    X_windows = []
    y_windows = []
    for i in range(n_samples):
        # Flatten the past lag_window days of X into one row
        window = X.iloc[i : i + lag_window].values.flatten()
        # Aggregate the future forecast_window days in y
        target = agg_func(y.iloc[i + lag_window : i + lag_window + forecast_window].values)

        X_windows.append(window)
        y_windows.append(target)

    return np.array(X_windows), np.array(y_windows)

# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------
def main():
    logging.info("Starting Forecasting Model Training Pipeline (14-day lag, 5-day forecast)")

    # -------------------------------
    # 1. Data Loading
    # -------------------------------
    data_dir = "Scripts/data_processing/processed_data/split_data_dir"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")

    logging.info(f"Loading split data from: {data_dir}")
    X_train = pd.read_csv(X_train_path, low_memory=False)
    X_test  = pd.read_csv(X_test_path,  low_memory=False)
    y_train = pd.read_csv(y_train_path, low_memory=False)
    y_test  = pd.read_csv(y_test_path,  low_memory=False)

    # Convert y columns to 1D arrays if needed
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    logging.info(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}, "
                 f"y_train={y_train.shape}, y_test={y_test.shape}")

    # -------------------------------
    # 2. Sliding Window Feature Creation
    # -------------------------------
    lag_window = 14
    forecast_window = 5
    agg_func = np.mean  # how we aggregate the forecast period (np.sum, np.mean, etc.)

    logging.info("Creating sliding windows for TRAIN set...")
    X_train_sw, y_train_sw = create_sliding_windows(X_train, y_train, lag_window, forecast_window, agg_func)

    logging.info("Creating sliding windows for TEST set...")
    X_test_sw, y_test_sw = create_sliding_windows(X_test, y_test, lag_window, forecast_window, agg_func)

    logging.info(f"Final sliding window shapes -> X_train_sw={X_train_sw.shape}, y_train_sw={y_train_sw.shape}, "
                 f"X_test_sw={X_test_sw.shape}, y_test_sw={y_test_sw.shape}")

    # -------------------------------
    # 3. Define Models
    # -------------------------------
    reporter = Reporter()
    models = {
        "KNN": KNNModel(n_neighbors=5),
        "Random Forest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
        "Logistic Regression": LogisticRegressionModel()
    }

    # -------------------------------
    # 4. Cross-Validation
    # -------------------------------
    for model_name, model_obj in models.items():
        logging.info(f"Performing cross-validation for {model_name}...")
        cv_metrics = cross_validate_model(model_obj, X_train_sw, y_train_sw, n_splits=5)
        logging.info(f"{model_name} (CV) metrics: {cv_metrics}")
        reporter.add_result(model_name + " (CV)", cv_metrics)

    # -------------------------------
    # 5. Train on Full Train Set & Evaluate on Test Set
    # -------------------------------
    for model_name, model_obj in models.items():
        logging.info(f"Training and evaluating {model_name} on test data...")
        model_obj.train(X_train_sw, y_train_sw)
        test_metrics = model_obj.evaluate(X_test_sw, y_test_sw)
        logging.info(f"{model_name} (Test) metrics: {test_metrics}")
        reporter.add_result(model_name + " (Test)", test_metrics)

    # -------------------------------
    # 6. Voting Classifier Ensemble
    # -------------------------------
    f1_threshold = 0.60  # example threshold
    logging.info(f"Filtering models with F1 >= {f1_threshold} from test metrics for ensemble...")
    selected_model_names = filter_models_by_threshold(reporter, threshold=f1_threshold)
    logging.info(f"Models selected for ensemble: {selected_model_names}")

    selected_models = {name: models[name] for name in selected_model_names if name in models}

    if selected_models:
        logging.info("Training Voting Classifier (soft voting) with selected models...")
        voting_clf = VotingClassifierCustom(selected_models, voting='soft')
        voting_clf.fit(X_train_sw, y_train_sw)

        # Evaluate on test set
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        voting_preds = voting_clf.predict(X_test_sw)
        voting_probs = voting_clf.predict_proba(X_test_sw)

        voting_metrics = {
            "accuracy": accuracy_score(y_test_sw, voting_preds),
            "precision": precision_score(y_test_sw, voting_preds),
            "recall": recall_score(y_test_sw, voting_preds),
            "f1_score": f1_score(y_test_sw, voting_preds),
            "roc_auc": roc_auc_score(y_test_sw, voting_probs[:,1])
        }
        reporter.add_result("Voting Classifier (Test)", voting_metrics)
        logging.info(f"Voting Classifier (Test) metrics: {voting_metrics}")

        # Save ensemble
        ensemble_path = "voting_model.pkl"
        joblib.dump(voting_clf, ensemble_path)
        logging.info(f"Voting Classifier saved to {ensemble_path}")
    else:
        logging.info("No models met the threshold; skipping Voting Classifier.")

    # -------------------------------
    # 7. Generate Final Report
    # -------------------------------
    final_report = reporter.generate_report("model_report.csv")
    logging.info("Final Model Performance Report:")
    logging.info("\n" + final_report.to_string())


if __name__ == "__main__":
    main()
