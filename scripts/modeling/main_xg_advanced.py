#!/usr/bin/env python3
import logging
import os
import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
import joblib

# For resampling & advanced approaches
from imblearn.combine import SMOTEENN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV

# GPU-Enabled Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data_with_dask(filepath):
    """Load CSV with Dask and convert to Pandas."""
    logging.info(f"[DEBUG] Checking existence of file: {filepath}")
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        sys.exit(f"Exiting due to missing file: {filepath}")
    logging.info(f"Loading data from {filepath} with Dask ...")
    df_dd = dd.read_csv(filepath, assume_missing=True)
    logging.info(f"Dask DataFrame loaded from {filepath}. Now converting to Pandas...")
    df = df_dd.compute()
    logging.info(f"Converted to Pandas. Shape: {df.shape}")
    return df

def threshold_predictions(probs, threshold=0.5):
    """Convert probability array to binary predictions using a specified threshold."""
    return (probs >= threshold).astype(int)

def main():
    logging.info("Starting advanced GPU pipeline (LightGBM, XGBoost, CatBoost) on SMOTEENN-resampled data...")

    # --------------------------------------------------------------------------
    # 1) Load Data
    # --------------------------------------------------------------------------
    data_dir = "scripts/data_processing/processed_data/split_data_dir"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")

    X_train_df = load_data_with_dask(X_train_path)
    X_test_df  = load_data_with_dask(X_test_path)
    y_train_df = load_data_with_dask(y_train_path)
    y_test_df  = load_data_with_dask(y_test_path)

    # --- New: Drop the 'date' column (if it exists) BEFORE converting to NumPy arrays.
    if 'date' in X_train_df.columns:
        logging.info("Dropping 'date' column from X_train DataFrame...")
        X_train_df = X_train_df.drop(columns=['date'])
    if 'date' in X_test_df.columns:
        logging.info("Dropping 'date' column from X_test DataFrame...")
        X_test_df = X_test_df.drop(columns=['date'])

    # Convert y dataframes to arrays (assume single column)
    y_train = y_train_df.iloc[:, 0].values
    y_test  = y_test_df.iloc[:, 0].values
    # Convert feature DataFrames to NumPy arrays now that 'date' is removed.
    X_train = X_train_df.values
    X_test  = X_test_df.values

    logging.info(f"Data shapes => X_train: {X_train.shape}, y_train: {y_train.shape}, "
                 f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --------------------------------------------------------------------------
    # 2) Resample with SMOTEENN (Hybrid oversampling + cleaning)
    # --------------------------------------------------------------------------
    logging.info("Applying SMOTEENN to training data for imbalance mitigation...")
    sme = SMOTEENN(random_state=42)
    try:
        X_train_res, y_train_res = sme.fit_resample(X_train, y_train)
    except Exception as e:
        logging.error(f"Error during SMOTEENN resampling: {e}")
        sys.exit("Resampling failed.")
    logging.info(f"After SMOTEENN => X_train_res: {X_train_res.shape}, y_train_res: {y_train_res.shape}")

    # --------------------------------------------------------------------------
    # 3) Define GPU-Enabled Models (LightGBM, XGBoost, CatBoost)
    # --------------------------------------------------------------------------
    # Note: For threshold tuning, we use a default threshold of 0.5; calibration steps are optional.
    # Also, we have commented out cross-validation for faster iteration.
    # --- LightGBM ---
    logging.info("Training LightGBM (GPU-enabled)...")
    lgb_clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        device_type='gpu'
    )
    lgb_clf.fit(X_train_res, y_train_res)
    lgb_probs = lgb_clf.predict_proba(X_test)[:, 1]
    lgb_preds = threshold_predictions(lgb_probs, threshold=0.5)
    lgb_metrics = {
        "accuracy": accuracy_score(y_test, lgb_preds),
        "precision": precision_score(y_test, lgb_preds, zero_division=0),
        "recall": recall_score(y_test, lgb_preds, zero_division=0),
        "f1_score": f1_score(y_test, lgb_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, lgb_probs),
    }
    logging.info(f"LightGBM (Test) metrics @ threshold=0.5 => {lgb_metrics}")

    logging.info("Calibrating LightGBM with isotonic regression (threshold=0.7 example)...")
    lgb_cal = CalibratedClassifierCV(base_estimator=lgb_clf, cv=3, method='isotonic')
    lgb_cal.fit(X_train_res, y_train_res)
    lgb_cal_probs = lgb_cal.predict_proba(X_test)[:, 1]
    lgb_cal_preds = threshold_predictions(lgb_cal_probs, threshold=0.7)
    lgb_cal_metrics = {
        "accuracy": accuracy_score(y_test, lgb_cal_preds),
        "precision": precision_score(y_test, lgb_cal_preds, zero_division=0),
        "recall": recall_score(y_test, lgb_cal_preds, zero_division=0),
        "f1_score": f1_score(y_test, lgb_cal_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, lgb_cal_probs),
    }
    logging.info(f"LightGBM + calibration, threshold=0.7 => {lgb_cal_metrics}")

    # --- XGBoost ---
    logging.info("Training XGBoost (GPU-enabled)...")
    xgb_clf = xgb.XGBClassifier(
        tree_method='gpu_hist',
        device='cuda:0',  # Use GPU
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        scale_pos_weight=np.sum(y_train_res == 0)/np.sum(y_train_res == 1),
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(X_train_res, y_train_res)
    xgb_probs = xgb_clf.predict_proba(X_test)[:, 1]
    xgb_preds = threshold_predictions(xgb_probs, threshold=0.5)
    xgb_metrics = {
        "accuracy": accuracy_score(y_test, xgb_preds),
        "precision": precision_score(y_test, xgb_preds, zero_division=0),
        "recall": recall_score(y_test, xgb_preds, zero_division=0),
        "f1_score": f1_score(y_test, xgb_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, xgb_probs),
    }
    logging.info(f"XGBoost (Test) metrics @ threshold=0.5 => {xgb_metrics}")

    logging.info("Calibrating XGBoost with isotonic regression (threshold=0.7 example)...")
    xgb_cal = CalibratedClassifierCV(base_estimator=xgb_clf, cv=3, method='isotonic')
    xgb_cal.fit(X_train_res, y_train_res)
    xgb_cal_probs = xgb_cal.predict_proba(X_test)[:, 1]
    xgb_cal_preds = threshold_predictions(xgb_cal_probs, threshold=0.7)
    xgb_cal_metrics = {
        "accuracy": accuracy_score(y_test, xgb_cal_preds),
        "precision": precision_score(y_test, xgb_cal_preds, zero_division=0),
        "recall": recall_score(y_test, xgb_cal_preds, zero_division=0),
        "f1_score": f1_score(y_test, xgb_cal_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, xgb_cal_probs),
    }
    logging.info(f"XGBoost + calibration, threshold=0.7 => {xgb_cal_metrics}")

    # --- CatBoost ---
    logging.info("Training CatBoost (GPU-enabled)...")
    cat_clf = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        task_type='GPU',    # GPU usage
        devices='0',        # Use GPU device 0
        eval_metric='Logloss',
        verbose=False
    )
    cat_clf.fit(X_train_res, y_train_res)
    cat_probs = cat_clf.predict_proba(X_test)[:, 1]
    cat_preds = threshold_predictions(cat_probs, threshold=0.5)
    cat_metrics = {
        "accuracy": accuracy_score(y_test, cat_preds),
        "precision": precision_score(y_test, cat_preds, zero_division=0),
        "recall": recall_score(y_test, cat_preds, zero_division=0),
        "f1_score": f1_score(y_test, cat_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, cat_probs),
    }
    logging.info(f"CatBoost (Test) metrics @ threshold=0.5 => {cat_metrics}")

    logging.info("Calibrating CatBoost with isotonic regression (threshold=0.7 example)...")
    cat_cal = CalibratedClassifierCV(base_estimator=cat_clf, cv=3, method='isotonic')
    cat_cal.fit(X_train_res, y_train_res)
    cat_cal_probs = cat_cal.predict_proba(X_test)[:, 1]
    cat_cal_preds = threshold_predictions(cat_cal_probs, threshold=0.7)
    cat_cal_metrics = {
        "accuracy": accuracy_score(y_test, cat_cal_preds),
        "precision": precision_score(y_test, cat_cal_preds, zero_division=0),
        "recall": recall_score(y_test, cat_cal_preds, zero_division=0),
        "f1_score": f1_score(y_test, cat_cal_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, cat_cal_probs),
    }
    logging.info(f"CatBoost + calibration, threshold=0.7 => {cat_cal_metrics}")

    # --------------------------------------------------------------------------
    # (Commented) Cross-Validation is skipped for now for faster iteration.
    # --------------------------------------------------------------------------
    # logging.info("Skipping cross-validation for now...")

    logging.info("Finished advanced pipeline with LightGBM, XGBoost, CatBoost (all GPU). Exiting.")

if __name__ == "__main__":
    main()
