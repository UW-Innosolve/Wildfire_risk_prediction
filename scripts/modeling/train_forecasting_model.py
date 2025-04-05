#!/usr/bin/env python3
import logging
import os
import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd
import joblib

from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# GPU-Enabled Libraries
import xgboost as xgb
from catboost import CatBoostClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------------------------------------------------------
# Sliding Window Function
# -----------------------------------------------------------------------------
def create_sliding_windows(X, y, lag_window=14, forecast_window=3, agg_func=np.max):
    """
    Creates forecasting samples using a sliding window approach.

    Each sample:
      - Input features: the flattened data of the previous 'lag_window' days in X
      - Target: aggregator (agg_func) over the next 'forecast_window' days in y
                e.g., np.max => if any day in next window is 1 => target=1

    Returns:
      X_new : np.array of shape (n_samples, lag_window * n_features)
      y_new : np.array of shape (n_samples,)
    """
    n_samples = len(y) - lag_window - forecast_window + 1
    if n_samples <= 0:
        raise ValueError("Not enough data to create the specified sliding windows.")

    X_list, y_list = [], []
    for i in range(n_samples):
        # Flatten the past 'lag_window' days into a single row
        feats = X.iloc[i : i+lag_window].values.flatten()
        # Aggregate the next 'forecast_window' days in y
        target = agg_func(y.iloc[i+lag_window : i+lag_window+forecast_window].values)
        X_list.append(feats)
        y_list.append(target)

    return np.array(X_list), np.array(y_list)


def load_data_with_dask(filepath):
    """Helper function to load CSV with Dask, then convert to Pandas."""
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        sys.exit(1)
    logging.info(f"Reading with Dask: {filepath}")
    df_dd = dd.read_csv(filepath, assume_missing=True)
    df = df_dd.compute()
    logging.info(f"Read shape {df.shape} from {filepath}")
    return df


def process_chunk(chunk_index, X_chunk, y_chunk, output_dir):
    """
    Process a single chunk of training data with SMOTEENN and save output.
    Returns the resampled X and y for this chunk.
    """
    logging.info(f"Processing chunk {chunk_index} with shape {X_chunk.shape} ...")
    sme = SMOTEENN(random_state=42)
    X_res, y_res = sme.fit_resample(X_chunk, y_chunk)
    # Save chunk to disk for checkpointing
    chunk_X_path = os.path.join(output_dir, f"X_train_resampled_chunk_{chunk_index}.csv")
    chunk_y_path = os.path.join(output_dir, f"y_train_resampled_chunk_{chunk_index}.csv")
    pd.DataFrame(X_res).to_csv(chunk_X_path, index=False)
    pd.DataFrame(y_res, columns=["y"]).to_csv(chunk_y_path, index=False)
    logging.info(f"Chunk {chunk_index} processed and saved to disk.")
    return X_res, y_res


def main():
    logging.info("Starting Forecasting + SMOTEENN pipeline (14-day lag, 3-day horizon, XGBoost & CatBoost).")

    # -------------------------------------------------------------------------
    # 1) Load Original Train/Test
    # -------------------------------------------------------------------------
    data_dir = "scripts/data_processing/processed_data/split_data_dir"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")

    X_train_df = load_data_with_dask(X_train_path)
    X_test_df  = load_data_with_dask(X_test_path)
    y_train_df = load_data_with_dask(y_train_path)
    y_test_df  = load_data_with_dask(y_test_path)

    # If y has shape (n,1), flatten
    if y_train_df.shape[1] == 1:
        y_train_df = y_train_df.iloc[:, 0]
    if y_test_df.shape[1] == 1:
        y_test_df = y_test_df.iloc[:, 0]

    # Drop 'date' if present
    if 'date' in X_train_df.columns:
        logging.info("Dropping 'date' from X_train_df.")
        X_train_df.drop(columns=['date'], inplace=True)
    if 'date' in X_test_df.columns:
        logging.info("Dropping 'date' from X_test_df.")
        X_test_df.drop(columns=['date'], inplace=True)

    logging.info(f"Shapes => X_train: {X_train_df.shape}, y_train: {y_train_df.shape}, "
                 f"X_test: {X_test_df.shape}, y_test: {y_test_df.shape}")

    # -------------------------------------------------------------------------
    # 2) Create Sliding Windows for Forecasting
    #    We do this on the original train/test, then apply SMOTEENN to the
    #    resulting (X_train_sw, y_train_sw) only.
    # -------------------------------------------------------------------------
    lag_window = 14
    forecast_window = 3
    agg_func = np.max  # aggregator to decide the future label

    logging.info(f"Creating sliding windows (lag={lag_window}, forecast={forecast_window}) on training set...")
    X_train_sw, y_train_sw = create_sliding_windows(X_train_df, y_train_df, lag_window, forecast_window, agg_func)
    logging.info(f"Creating sliding windows on test set (no SMOTE on test) ...")
    X_test_sw, y_test_sw = create_sliding_windows(X_test_df, y_test_df, lag_window, forecast_window, agg_func)

    logging.info(f"Windowed shapes => X_train_sw={X_train_sw.shape}, y_train_sw={y_train_sw.shape}, "
                 f"X_test_sw={X_test_sw.shape}, y_test_sw={y_test_sw.shape}")

    # -------------------------------------------------------------------------
    # 3) Apply Chunked SMOTEENN to the Sliding-Window Train Data
    # -------------------------------------------------------------------------
    n_rows = X_train_sw.shape[0]
    n_chunks = 5
    chunk_size = int(np.ceil(n_rows / n_chunks))
    sampled_dir = os.path.join("scripts", "data_processing", "processed_data", "forecast_smoteenn_dir")
    os.makedirs(sampled_dir, exist_ok=True)

    X_chunks, y_chunks = [], []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, n_rows)
        logging.info(f"Resampling chunk {i+1}/{n_chunks}: rows {start} to {end}")
        X_chunk = X_train_sw[start:end]
        y_chunk = y_train_sw[start:end]
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

    logging.info(f"Final SMOTEENN-resampled train shape => X_train_res: {X_train_res.shape}, y_train_res: {y_train_res.shape}")
    # We do NOT resample the test set, so it remains (X_test_sw, y_test_sw).

    # -------------------------------------------------------------------------
    # 4) Train XGBoost on (X_train_res, y_train_res), Evaluate on Test
    # -------------------------------------------------------------------------
    logging.info("Training XGBoost (GPU if available) on resampled forecasting data...")
    # For XGBoost 2.0+ => 'device': 'cuda', 'tree_method': 'hist'
    xgb_clf = xgb.XGBClassifier(
        tree_method='hist',
        device='cuda',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        eval_metric='logloss'
    )
    xgb_clf.fit(X_train_res, y_train_res)

    # Evaluate at threshold=0.5
    xgb_probs_05 = xgb_clf.predict_proba(X_test_sw)[:, 1]
    xgb_preds_05 = (xgb_probs_05 >= 0.5).astype(int)
    xgb_05_metrics = {
        "accuracy": accuracy_score(y_test_sw, xgb_preds_05),
        "precision": precision_score(y_test_sw, xgb_preds_05, zero_division=0),
        "recall":    recall_score(y_test_sw, xgb_preds_05, zero_division=0),
        "f1_score":  f1_score(y_test_sw, xgb_preds_05, zero_division=0),
        "roc_auc":   roc_auc_score(y_test_sw, xgb_probs_05),
    }
    logging.info(f"XGBoost (threshold=0.5) => {xgb_05_metrics}")

    # Evaluate at threshold=0.7
    xgb_preds_07 = (xgb_probs_05 >= 0.7).astype(int)
    xgb_07_metrics = {
        "accuracy": accuracy_score(y_test_sw, xgb_preds_07),
        "precision": precision_score(y_test_sw, xgb_preds_07, zero_division=0),
        "recall":    recall_score(y_test_sw, xgb_preds_07, zero_division=0),
        "f1_score":  f1_score(y_test_sw, xgb_preds_07, zero_division=0),
        "roc_auc":   roc_auc_score(y_test_sw, xgb_probs_05),
    }
    logging.info(f"XGBoost (threshold=0.7) => {xgb_07_metrics}")

    # Optional: XGBoost calibration
    logging.info("Calibrating XGBoost with isotonic regression (threshold=0.7 example)...")
    xgb_cal = CalibratedClassifierCV(base_estimator=xgb_clf, cv=3, method='isotonic')
    xgb_cal.fit(X_train_res, y_train_res)
    xgb_cal_probs = xgb_cal.predict_proba(X_test_sw)[:, 1]
    xgb_cal_preds = (xgb_cal_probs >= 0.7).astype(int)
    xgb_cal_metrics = {
        "accuracy": accuracy_score(y_test_sw, xgb_cal_preds),
        "precision": precision_score(y_test_sw, xgb_cal_preds, zero_division=0),
        "recall": recall_score(y_test_sw, xgb_cal_preds, zero_division=0),
        "f1_score": f1_score(y_test_sw, xgb_cal_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test_sw, xgb_cal_probs),
    }
    logging.info(f"XGBoost + calibration (threshold=0.7) => {xgb_cal_metrics}")

    # -------------------------------------------------------------------------
    # 5) Train CatBoost
    # -------------------------------------------------------------------------
    logging.info("Training CatBoost (GPU) on resampled forecasting data...")
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

    # Evaluate at threshold=0.5
    cat_probs_05 = cat_clf.predict_proba(X_test_sw)[:, 1]
    cat_preds_05 = (cat_probs_05 >= 0.5).astype(int)
    cat_05_metrics = {
        "accuracy":  accuracy_score(y_test_sw, cat_preds_05),
        "precision": precision_score(y_test_sw, cat_preds_05, zero_division=0),
        "recall":    recall_score(y_test_sw, cat_preds_05, zero_division=0),
        "f1_score":  f1_score(y_test_sw, cat_preds_05, zero_division=0),
        "roc_auc":   roc_auc_score(y_test_sw, cat_probs_05),
    }
    logging.info(f"CatBoost (threshold=0.5) => {cat_05_metrics}")

    # Evaluate at threshold=0.7
    cat_preds_07 = (cat_probs_05 >= 0.7).astype(int)
    cat_07_metrics = {
        "accuracy": accuracy_score(y_test_sw, cat_preds_07),
        "precision": precision_score(y_test_sw, cat_preds_07, zero_division=0),
        "recall": recall_score(y_test_sw, cat_preds_07, zero_division=0),
        "f1_score": f1_score(y_test_sw, cat_preds_07, zero_division=0),
        "roc_auc": roc_auc_score(y_test_sw, cat_probs_05),
    }
    logging.info(f"CatBoost (threshold=0.7) => {cat_07_metrics}")

    # Optional: CatBoost calibration
    logging.info("Calibrating CatBoost (isotonic, threshold=0.7)...")
    cat_cal = CalibratedClassifierCV(base_estimator=cat_clf, cv=3, method='isotonic')
    cat_cal.fit(X_train_res, y_train_res)
    cat_cal_probs = cat_cal.predict_proba(X_test_sw)[:, 1]
    cat_cal_preds = (cat_cal_probs >= 0.7).astype(int)
    cat_cal_metrics = {
        "accuracy": accuracy_score(y_test_sw, cat_cal_preds),
        "precision": precision_score(y_test_sw, cat_cal_preds, zero_division=0),
        "recall": recall_score(y_test_sw, cat_cal_preds, zero_division=0),
        "f1_score": f1_score(y_test_sw, cat_cal_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test_sw, cat_cal_probs),
    }
    logging.info(f"CatBoost + calibration (threshold=0.7) => {cat_cal_metrics}")

    # -------------------------------------------------------------------------
    # (Cross-validation commented out for brevity)
    # -------------------------------------------------------------------------
    logging.info("Finished Forecasting + SMOTEENN pipeline. Exiting now.")
    sys.exit(0)


if __name__ == "__main__":
    main()
