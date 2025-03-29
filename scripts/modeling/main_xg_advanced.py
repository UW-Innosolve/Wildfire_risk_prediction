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

def process_chunk(chunk_index, X_chunk, y_chunk, output_dir):
    """
    Process a single chunk of training data with SMOTEENN and save its output.
    Returns the resampled X and y for this chunk.
    """
    from imblearn.combine import SMOTEENN  # local import to ensure checkpointing
    logging.info(f"Processing chunk {chunk_index} with shape {X_chunk.shape}...")
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_resample(X_chunk, y_chunk)
    chunk_X_path = os.path.join(output_dir, f"X_train_resampled_chunk_{chunk_index}.csv")
    chunk_y_path = os.path.join(output_dir, f"y_train_resampled_chunk_{chunk_index}.csv")
    pd.DataFrame(X_res).to_csv(chunk_X_path, index=False)
    pd.DataFrame(y_res, columns=["y"]).to_csv(chunk_y_path, index=False)
    logging.info(f"Chunk {chunk_index} processed and saved.")
    return X_res, y_res

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

    # --- Drop the 'date' column if it exists ---
    if 'date' in X_train_df.columns:
        logging.info("Dropping 'date' column from X_train DataFrame...")
        X_train_df = X_train_df.drop(columns=['date'])
    if 'date' in X_test_df.columns:
        logging.info("Dropping 'date' column from X_test DataFrame...")
        X_test_df = X_test_df.drop(columns=['date'])

    # Convert to NumPy arrays (assume y is a single column)
    y_train = y_train_df.iloc[:, 0].values
    y_test  = y_test_df.iloc[:, 0].values
    X_train = X_train_df.values
    X_test  = X_test_df.values

    logging.info(f"Data shapes => X_train: {X_train.shape}, y_train: {y_train.shape}, "
                 f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # --------------------------------------------------------------------------
    # 2) Resample with SMOTEENN in Chunks (for checkpointing and speed)
    # --------------------------------------------------------------------------
    # We'll split the training data into chunks and process each with SMOTEENN.
    n_rows = X_train.shape[0]
    n_chunks = 5  # Adjust based on your available resources
    chunk_size = int(np.ceil(n_rows / n_chunks))
    sampled_dir = os.path.join("scripts", "data_processing", "processed_data", "split_data_dir_sampled")
    os.makedirs(sampled_dir, exist_ok=True)
    
    X_chunks = []
    y_chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, n_rows)
        logging.info(f"Processing chunk {i+1}/{n_chunks}: rows {start} to {end}")
        X_chunk = X_train[start:end]
        y_chunk = y_train[start:end]
        chunk_X_path = os.path.join(sampled_dir, f"X_train_resampled_chunk_{i+1}.csv")
        chunk_y_path = os.path.join(sampled_dir, f"y_train_resampled_chunk_{i+1}.csv")
        if os.path.exists(chunk_X_path) and os.path.exists(chunk_y_path):
            logging.info(f"Loading resampled chunk {i+1} from disk...")
            X_res_chunk = pd.read_csv(chunk_X_path).values
            y_res_chunk = pd.read_csv(chunk_y_path)["y"].values
        else:
            X_res_chunk, y_res_chunk = process_chunk(i+1, X_chunk, y_chunk, sampled_dir)
        X_chunks.append(X_res_chunk)
        y_chunks.append(y_res_chunk)
    X_train_res = np.vstack(X_chunks)
    y_train_res = np.concatenate(y_chunks)
    logging.info(f"Final resampled training data shape: X_train_res: {X_train_res.shape}, y_train_res: {y_train_res.shape}")

    # Save test data as well for consistency
    pd.DataFrame(X_test).to_csv(os.path.join(sampled_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_test, columns=["y"]).to_csv(os.path.join(sampled_dir, "y_test.csv"), index=False)
    logging.info(f"Test data saved to {sampled_dir}")

    # --------------------------------------------------------------------------
    # 3) Define GPU-Enabled Models (LightGBM, XGBoost, CatBoost)
    # (Cross-validation is commented out for now for faster iteration)
    # --------------------------------------------------------------------------
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
        device='cuda:0',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        scale_pos_weight=np.sum(y_train_res == 0) / np.sum(y_train_res == 1),
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
        task_type='GPU',
        devices='0',
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
    # (Commented) Cross-Validation is skipped for faster iteration.
    # --------------------------------------------------------------------------
    # logging.info("Skipping cross-validation for now...")

    logging.info("Finished advanced pipeline with LightGBM, XGBoost, CatBoost (all GPU). Exiting.")

if __name__ == "__main__":
    main()
