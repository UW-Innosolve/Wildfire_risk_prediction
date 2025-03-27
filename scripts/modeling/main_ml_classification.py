import logging
import os
import joblib
import dask.dataframe as dd
import pandas as pd

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

# === Optional: For specifying parallel or chunking parameters ===
from sklearn.model_selection import StratifiedKFold, cross_val_score

# -----------------------------------------------------------------------------
# Configure logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------------------------------------------------------
# Helper: Load data using Dask
# -----------------------------------------------------------------------------
def load_data_with_dask(filepath):
    logging.info(f"Loading data from {filepath} with Dask ...")
    df = dd.read_csv(filepath, assume_missing=True)
    logging.info(f"Dask DataFrame loaded from {filepath}")
    return df

# -----------------------------------------------------------------------------
# Main pipeline using Dask with merging to align X and y
# -----------------------------------------------------------------------------
def main():
    logging.info("Starting Wildfire Prediction Classification Pipeline using Dask for loading/filtering")

    # -------------------------------
    # 1. Data Loading with Dask
    # -------------------------------
    data_dir = "Scripts/data_processing/processed_data/split_data_dir"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")
    
    # Load features and targets; reset indices.
    X_train_dd = load_data_with_dask(X_train_path).reset_index(drop=True)
    X_test_dd  = load_data_with_dask(X_test_path).reset_index(drop=True)
    y_train_dd = load_data_with_dask(y_train_path).reset_index(drop=True)
    y_test_dd  = load_data_with_dask(y_test_path).reset_index(drop=True)
    
    # Rename target columns to 'y'
    y_train_dd = y_train_dd.rename(columns={y_train_dd.columns[0]: 'y'})
    y_test_dd = y_test_dd.rename(columns={y_test_dd.columns[0]: 'y'})
    
    logging.info("Initial Dask DataFrames loaded.")
    
    # -------------------------------
    # 1a. Create an index column and merge X and y to ensure alignment
    # -------------------------------
    X_train_dd = X_train_dd.assign(idx=lambda df: df.index)
    y_train_dd = y_train_dd.assign(idx=lambda df: df.index)
    df_train_dd = dd.merge(X_train_dd, y_train_dd, on="idx")
    
    X_test_dd = X_test_dd.assign(idx=lambda df: df.index)
    y_test_dd = y_test_dd.assign(idx=lambda df: df.index)
    df_test_dd = dd.merge(X_test_dd, y_test_dd, on="idx")
    
    # -------------------------------
    # (Removed code that filters rows by date for specific months)
    # -------------------------------
    # If you still want to drop the 'date' column (assuming you don't need it):
    # if 'date' in df_train_dd.columns:
    #     df_train_dd = df_train_dd.drop(columns=['date'])
    #     df_test_dd = df_test_dd.drop(columns=['date'])

    # -------------------------------
    # 1c. Convert Dask DataFrames to pandas DataFrames and split back into X and y
    # -------------------------------
    logging.info("Converting Dask DataFrames to pandas DataFrames...")
    df_train = df_train_dd.compute()
    df_test = df_test_dd.compute()
    
    # Drop the merging index column ("idx")
    df_train = df_train.drop(columns=['idx'])
    df_test = df_test.drop(columns=['idx'])
    
    # Split the merged DataFrame back into features (X) and target (y)
    y_train = df_train.pop('y')
    y_test = df_test.pop('y')
    X_train = df_train
    X_test = df_test
    if 'date' in X_train.columns:
     X_train.drop(columns=['date'], inplace=True)
    if 'date' in X_test.columns:
     X_test.drop(columns=['date'], inplace=True)
    logging.info(f"Final shapes -> X_train={X_train.shape}, X_test={X_test.shape}, "
                 f"y_train={y_train.shape}, y_test={y_test.shape}")
    
    # -------------------------------
    # 2. Define Models
    # -------------------------------
    reporter = Reporter()
    models = {
        "KNN": KNNModel(n_neighbors=5),
        "Random Forest": RandomForestModel(
            params={'n_estimators': 100,
                    'max_depth': None,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1}
        ),
        "XGBoost": XGBoostModel(
            params={'learning_rate': 0.1,
                    'max_depth': 6,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'scale_pos_weight': 5,
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'nthread': -1}
        ),
        "Logistic Regression": LogisticRegressionModel(
            params={'C': 1.0,
                    'solver': 'liblinear',
                    'class_weight': 'balanced',
                    'max_iter': 1000}
        ),
    }
    
    # -------------------------------
    # 3. Cross-Validation on Training Set
    # -------------------------------
    logging.info("Starting cross-validation on the training set...")
    for model_name, model_obj in models.items():
        logging.info(f"Cross-validating model: {model_name}")
        cv_metrics = cross_validate_model(model_obj, X_train, y_train)
        logging.info(f"{model_name} (CV) metrics: {cv_metrics}")
        reporter.add_result(model_name + " (CV)", cv_metrics)
    
    # -------------------------------
    # 4. Train on Full Training Set & Evaluate on Test Set
    # -------------------------------
    logging.info("Training models on full training set and evaluating on test set...")
    for model_name, model_obj in models.items():
        logging.info(f"Training {model_name}...")
        model_obj.train(X_train, y_train)
        
        logging.info(f"Evaluating {model_name} on test data...")
        test_metrics = model_obj.evaluate(X_test, y_test)
        logging.info(f"{model_name} (Test) metrics: {test_metrics}")
        reporter.add_result(model_name + " (Test)", test_metrics)
    
    # -------------------------------
    # 5. Build a Voting Classifier
    # -------------------------------
    f1_threshold = 0.60  # example threshold
    logging.info(f"Filtering models with F1 >= {f1_threshold} (Test) to build ensemble...")
    selected_model_names = filter_models_by_threshold(reporter, threshold=f1_threshold)
    logging.info(f"Models selected for ensemble: {selected_model_names}")
    
    selected_models = {name: models[name] for name in selected_model_names if name in models}
    
    if not selected_models:
        logging.info("No models met the threshold; skipping Voting Classifier.")
    else:
        voting_clf = VotingClassifierCustom(
            selected_models,  # dict of {model_name: model_obj}
            voting='soft'
        )
        logging.info("Fitting the Voting Classifier on the full training set...")
        voting_clf.fit(X_train, y_train)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        voting_preds = voting_clf.predict(X_test)
        voting_probs = voting_clf.predict_proba(X_test)[:, 1]  # probability for positive class
        
        voting_metrics = {
            "accuracy":  accuracy_score(y_test, voting_preds),
            "precision": precision_score(y_test, voting_preds),
            "recall":    recall_score(y_test, voting_preds),
            "f1_score":  f1_score(y_test, voting_preds),
            "roc_auc":   roc_auc_score(y_test, voting_probs)
        }
        logging.info(f"Voting Classifier (Test) metrics: {voting_metrics}")
        reporter.add_result("Voting Classifier (Test)", voting_metrics)
        
        ensemble_path = "voting_model.pkl"
        joblib.dump(voting_clf, ensemble_path)
        logging.info(f"Voting Classifier saved to: {ensemble_path}")
    
    # -------------------------------
    # 6. Generate Final Performance Report
    # -------------------------------
    final_report = reporter.generate_report("model_report.csv")
    logging.info("Final Model Performance Report:")
    logging.info("\n" + final_report.to_string())
    
    logging.info("Done! Exiting pipeline.")

if __name__ == "__main__":
    main()
