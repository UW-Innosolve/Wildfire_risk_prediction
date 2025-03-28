#!/usr/bin/env python3
import logging
import os
import sys
import joblib
import dask.dataframe as dd
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

# For SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_with_dask(filepath):
    logging.info(f"[DEBUG] Checking existence of file: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        sys.exit(f"Exiting due to missing file: {filepath}")
    logging.info(f"Loading data from {filepath} with Dask ...")
    df = dd.read_csv(filepath, assume_missing=True)
    logging.info(f"Dask DataFrame loaded from {filepath}")
    return df

def main():
    logging.info("Starting training pipeline with SMOTE for class imbalance")

    # -------------------------------
    # Data Loading
    # -------------------------------
    data_dir = "scripts/data_processing/processed_data/split_data_dir"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")

    logging.info(f"[DEBUG] X_train_path: {X_train_path}")
    logging.info(f"[DEBUG] X_test_path: {X_test_path}")
    logging.info(f"[DEBUG] y_train_path: {y_train_path}")
    logging.info(f"[DEBUG] y_test_path: {y_test_path}")

    X_train_dd = load_data_with_dask(X_train_path).reset_index(drop=True)
    X_test_dd  = load_data_with_dask(X_test_path).reset_index(drop=True)
    y_train_dd = load_data_with_dask(y_train_path).reset_index(drop=True)
    y_test_dd  = load_data_with_dask(y_test_path).reset_index(drop=True)
    
    logging.info("Renaming target columns to 'y' if needed...")
    y_train_dd = y_train_dd.rename(columns={y_train_dd.columns[0]: 'y'})
    y_test_dd = y_test_dd.rename(columns={y_test_dd.columns[0]: 'y'})
    logging.info("Initial Dask DataFrames loaded.")

    # -------------------------------
    # Merge X and y for alignment and compute Pandas DataFrames
    # -------------------------------
    X_train_dd = X_train_dd.assign(idx=lambda df: df.index)
    y_train_dd = y_train_dd.assign(idx=lambda df: df.index)
    df_train_dd = dd.merge(X_train_dd, y_train_dd, on="idx")
    
    X_test_dd = X_test_dd.assign(idx=lambda df: df.index)
    y_test_dd = y_test_dd.assign(idx=lambda df: df.index)
    df_test_dd = dd.merge(X_test_dd, y_test_dd, on="idx")
    
    logging.info("Computing Dask DataFrames to Pandas for training set...")
    df_train = df_train_dd.compute()
    logging.info(f"[DEBUG] df_train shape: {df_train.shape}")

    logging.info("Computing Dask DataFrames to Pandas for testing set...")
    df_test = df_test_dd.compute()
    logging.info(f"[DEBUG] df_test shape: {df_test.shape}")
    
    df_train = df_train.drop(columns=['idx'])
    df_test = df_test.drop(columns=['idx'])
    
    logging.info("Splitting merged DataFrames into features (X) and target (y)...")
    y_train = df_train.pop('y')
    y_test = df_test.pop('y')
    X_train = df_train
    X_test = df_test

    if 'date' in X_train.columns:
        logging.info("Dropping 'date' column from X_train")
        X_train.drop(columns=['date'], inplace=True)
    if 'date' in X_test.columns:
        logging.info("Dropping 'date' column from X_test")
        X_test.drop(columns=['date'], inplace=True)

    logging.info(f"Final shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}, y_test={y_test.shape}")
    
    if X_train.empty or y_train.empty:
        logging.error("Training data or labels are empty. Exiting.")
        sys.exit("No data to train on.")

    # -------------------------------
    # Convert to contiguous NumPy arrays
    # -------------------------------
    logging.info("Converting DataFrames to contiguous NumPy arrays...")
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test  = X_test.apply(pd.to_numeric, errors='coerce')
    X_train = np.ascontiguousarray(X_train.values)
    X_test  = np.ascontiguousarray(X_test.values)
    y_train = np.ascontiguousarray(y_train.values)
    y_test  = np.ascontiguousarray(y_test.values)
    logging.info(f"[DEBUG] After conversion: X_train shape = {X_train.shape}, X_test shape = {X_test.shape}")
    logging.info(f"[DEBUG] y_train shape = {y_train.shape}, y_test shape = {y_test.shape}")

    # -------------------------------
    # Apply SMOTE to address class imbalance (only on training data)
    # -------------------------------
    logging.info("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    logging.info(f"After SMOTE: X_train_res shape = {X_train_res.shape}, y_train_res shape = {y_train_res.shape}")

    # Save resampled data to CSV files in a new folder
    sampled_dir = os.path.join("scripts", "data_processing", "processed_data", "split_data_dir_sampled")
    os.makedirs(sampled_dir, exist_ok=True)
    pd.DataFrame(X_train_res).to_csv(os.path.join(sampled_dir, "X_train_resampled.csv"), index=False)
    pd.DataFrame(y_train_res, columns=["y"]).to_csv(os.path.join(sampled_dir, "y_train_resampled.csv"), index=False)
    # Also save the test data (unchanged) for consistency
    pd.DataFrame(X_test).to_csv(os.path.join(sampled_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_test, columns=["y"]).to_csv(os.path.join(sampled_dir, "y_test.csv"), index=False)
    logging.info(f"Resampled and test data saved to {sampled_dir}")

    # -------------------------------
    # Define Models with Parallel/GPU Settings
    # -------------------------------
    reporter = Reporter()
    from model_classes.fb_approx_knn import ApproxKNNModel  # Assuming you have an approximate KNN class implemented
    models = {
        "Approx KNN": ApproxKNNModel(n_neighbors=23, n_trees=10, metric='euclidean'),
        "Random Forest": RandomForestModel(
            params={
                'n_estimators': 100,
                'max_depth': None,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        ),
        "XGBoost": XGBoostModel(
            params={
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'scale_pos_weight': (np.sum(y_train_res == 0) / np.sum(y_train_res == 1)),
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'tree_method': 'gpu_hist',  # GPU acceleration
                'gpu_id': 0
            }
        ),
        "Logistic Regression": LogisticRegressionModel(
            params={
                'C': 1.0,
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'max_iter': 1000
            }
        ),
    }
    
    # -------------------------------
    # Cross-Validation on Resampled Training Set (Optional: comment out for initial testing)
    # -------------------------------
    # logging.info("Starting cross-validation on SMOTE-resampled training set...")
    # for model_name, model_obj in models.items():
    #     logging.info(f"Cross-validating model: {model_name}")
    #     cv_metrics = cross_validate_model(model_obj, X_train_res, y_train_res)
    #     logging.info(f"{model_name} (CV) metrics: {cv_metrics}")
    #     reporter.add_result(model_name + " (CV)", cv_metrics)
    
    # -------------------------------
    # Train on Full Resampled Training Set & Evaluate on Test Set
    # -------------------------------
    logging.info("Training models on full SMOTE-resampled training set and evaluating on test set...")
    for model_name, model_obj in models.items():
        logging.info(f"Training {model_name}...")
        model_obj.train(X_train_res, y_train_res)
        
        logging.info(f"Evaluating {model_name} on test data...")
        test_metrics = model_obj.evaluate(X_test, y_test)
        logging.info(f"{model_name} (Test) metrics: {test_metrics}")
        reporter.add_result(model_name + " (Test)", test_metrics)
    
    # -------------------------------
    # Build a Voting Classifier Ensemble
    # -------------------------------
    f1_threshold = 0.60  # example threshold
    logging.info(f"Filtering models with F1 >= {f1_threshold} (Test) to build ensemble...")
    selected_model_names = filter_models_by_threshold(reporter, threshold=f1_threshold)
    logging.info(f"Models selected for ensemble: {selected_model_names}")
    
    selected_models = {name: models[name] for name in selected_model_names if name in models}
    
    if not selected_models:
        logging.info("No models met the threshold; skipping Voting Classifier.")
    else:
        logging.info("Fitting the Voting Classifier on the full SMOTE-resampled training set...")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        from model_classes.voting_classifier import VotingClassifierCustom
        
        voting_clf = VotingClassifierCustom(selected_models, voting='soft')
        voting_clf.fit(X_train_res, y_train_res)
        
        voting_preds = voting_clf.predict(X_test)
        voting_probs = voting_clf.predict_proba(X_test)[:, 1]
        
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
    # Generate Final Performance Report
    # -------------------------------
    final_report = reporter.generate_report("model_report.csv")
    logging.info("Final Model Performance Report:")
    logging.info("\n" + final_report.to_string())
    
    logging.info("Done! Exiting pipeline.")

if __name__ == "__main__":
    main()
