#!/usr/bin/env python3
import logging
import os
import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd

from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
from catboost import CatBoostClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

def create_windows_for_chunk(
    X_slice, y_slice,
    chunk_start_idx,  # where X_slice starts in the global dataset
    actual_start,     # actual chunk boundaries (no overlap region)
    actual_end,       
    lag_window=14,
    forecast_window=3,
    agg_func=np.max
):
    """
    Create sliding windows for the chunk with overlap. Only keep windows whose
    'start day' is in [actual_start, actual_end - lag_window - forecast_window].
    Returns (X_win, y_win).
    """
    # We'll define local index 0 => chunk_start_idx in the global dataset
    # The slice length is len(X_slice)
    n = len(X_slice)
    if n < (lag_window + forecast_window):
        return np.empty((0, X_slice.shape[1]*lag_window)), np.empty((0,))

    X_list, y_list = [], []
    max_local_start = n - (lag_window + forecast_window)
    for local_start in range(max_local_start + 1):
        global_start = chunk_start_idx + local_start
        # Keep only if global_start is within [actual_start, actual_end - lag - forecast]
        if global_start < actual_start or global_start > (actual_end - lag_window - forecast_window):
            continue

        # Flatten features for the past lag_window days
        feats = X_slice.iloc[local_start:local_start+lag_window].values.flatten()
        # aggregator for next forecast_window days
        label_val = agg_func(y_slice.iloc[local_start+lag_window : local_start+lag_window+forecast_window].values)

        X_list.append(feats)
        y_list.append(label_val)

    if not X_list:
        return np.empty((0, X_slice.shape[1]*lag_window)), np.empty((0,))
    return np.vstack(X_list), np.array(y_list)


def main():
    logging.info("Starting chunk-based sliding windows + SMOTEENN for large forecast dataset (no single pass).")

    # -- 1) Load Original Train/Test
    data_dir = "scripts/data_processing/processed_data/split_data_dir"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")

    X_train_df = load_data_with_dask(X_train_path)
    X_test_df  = load_data_with_dask(X_test_path)
    y_train_df = load_data_with_dask(y_train_path)
    y_test_df  = load_data_with_dask(y_test_path)

    # Flatten y if shape (n,1)
    if y_train_df.shape[1] == 1:
        y_train_df = y_train_df.iloc[:,0]
    if y_test_df.shape[1] == 1:
        y_test_df = y_test_df.iloc[:,0]

    # Drop 'date' col if present
    if 'date' in X_train_df.columns:
        logging.info("Dropping 'date' from X_train_df.")
        X_train_df.drop(columns=['date'], inplace=True)
    if 'date' in X_test_df.columns:
        logging.info("Dropping 'date' from X_test_df.")
        X_test_df.drop(columns=['date'], inplace=True)

    n_train = len(X_train_df)
    logging.info(f"Shapes => X_train: {X_train_df.shape}, y_train: {y_train_df.shape}, "
                 f"X_test: {X_test_df.shape}, y_test: {y_test_df.shape}")

    lag_window = 14
    forecast_window = 3
    agg_func = np.max

    # -- 2) Chunk-based creation of windows + SMOTEENN
    chunk_size = 500_000
    overlap = lag_window + forecast_window - 1  # how many rows overlap

    out_dir = "scripts/data_processing/processed_data/forecast_chunks"
    os.makedirs(out_dir, exist_ok=True)

    # We'll store the processed chunk files, then combine
    chunked_files = []
    start_idx = 0
    chunk_id = 1

    while start_idx < n_train:
        # actual chunk boundaries (no overlap)
        chunk_end_idx = min(start_idx + chunk_size, n_train)

        # We'll define an extended slice with overlap
        extended_start = max(0, start_idx - overlap)
        extended_end   = min(n_train, chunk_end_idx + overlap)

        logging.info(f"[Chunk {chunk_id}] Extended slice: [{extended_start}, {extended_end}) Overlap to handle windows.")
        logging.info(f"[Chunk {chunk_id}] Actual chunk coverage: [{start_idx}, {chunk_end_idx})")

        X_slice = X_train_df.iloc[extended_start:extended_end]
        y_slice = y_train_df.iloc[extended_start:extended_end]

        # create sliding windows for this chunk
        X_chunk_win, y_chunk_win = create_windows_for_chunk(
            X_slice, y_slice,
            chunk_start_idx=extended_start,
            actual_start=start_idx,
            actual_end=chunk_end_idx,
            lag_window=lag_window,
            forecast_window=forecast_window,
            agg_func=agg_func
        )
        logging.info(f"[Chunk {chunk_id}] Created {X_chunk_win.shape[0]} windows after filtering boundaries.")

        if X_chunk_win.shape[0] == 0:
            # no windows, skip
            logging.info(f"[Chunk {chunk_id}] No valid windows, skipping SMOTEENN.")
        else:
            # apply SMOTEENN
            logging.info(f"[Chunk {chunk_id}] Applying SMOTEENN to chunk's windows...")
            sme = SMOTEENN(random_state=42)
            X_res, y_res = sme.fit_resample(X_chunk_win, y_chunk_win)
            logging.info(f"[Chunk {chunk_id}] After SMOTEENN => {X_res.shape[0]} samples")

            # Save chunk
            cx = os.path.join(out_dir, f"train_chunk_{chunk_id}_X.csv")
            cy = os.path.join(out_dir, f"train_chunk_{chunk_id}_y.csv")
            pd.DataFrame(X_res).to_csv(cx, index=False)
            pd.DataFrame(y_res, columns=["y"]).to_csv(cy, index=False)
            chunked_files.append((cx, cy))
            logging.info(f"[Chunk {chunk_id}] Saved chunk => {cx}, {cy}")

        start_idx = chunk_end_idx
        chunk_id += 1

    # combine chunked SMOTE data
    logging.info("Combining all chunked SMOTEENN CSVs...")
    all_X_list, all_y_list = [], []
    for (cx, cy) in chunked_files:
        Xp = pd.read_csv(cx)
        yp = pd.read_csv(cy)["y"]
        all_X_list.append(Xp)
        all_y_list.append(yp)
    X_train_res = pd.concat(all_X_list, ignore_index=True).values
    y_train_res = pd.concat(all_y_list, ignore_index=True).values
    logging.info(f"Final TRAIN shape after chunk-based windows + SMOTEENN => {X_train_res.shape}, {y_train_res.shape}")

    # -- 3) Create sliding windows for test set (no SMOTE).
    # If test set is smaller, we can do a single pass
    # Or chunk it similarly if needed
    logging.info("Creating sliding windows for the test set (single pass).")
    def create_sw_test(X_df, y_df, lag=14, forecast=3, aggregator=np.max):
        N = len(X_df)
        max_start = N - lag - forecast
        if max_start < 0:
            return np.empty((0, X_df.shape[1]*lag)), np.empty((0,))
        X_list, y_list = [], []
        for i in range(max_start + 1):
            feats = X_df.iloc[i:i+lag].values.flatten()
            val   = aggregator(y_df.iloc[i+lag:i+lag+forecast].values)
            X_list.append(feats)
            y_list.append(val)
        return np.vstack(X_list), np.array(y_list)

    X_test_sw, y_test_sw = create_sw_test(X_test_df, y_test_df, lag_window, forecast_window, np.max)
    logging.info(f"Final TEST window shape => {X_test_sw.shape}, {y_test_sw.shape}")

    # -- 4) Train XGBoost on chunk-based data
    logging.info("Training XGBoost on final chunk-based windows (GPU)...")
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

    # Evaluate threshold=0.5
    xgb_probs = xgb_clf.predict_proba(X_test_sw)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)
    xgb_metrics = {
        "accuracy": accuracy_score(y_test_sw, xgb_preds),
        "precision": precision_score(y_test_sw, xgb_preds, zero_division=0),
        "recall": recall_score(y_test_sw, xgb_preds, zero_division=0),
        "f1_score": f1_score(y_test_sw, xgb_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test_sw, xgb_probs),
    }
    logging.info(f"XGBoost (threshold=0.5) => {xgb_metrics}")

    # -- CatBoost
    logging.info("Training CatBoost (GPU)...")
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
    cat_probs = cat_clf.predict_proba(X_test_sw)[:, 1]
    cat_preds = (cat_probs >= 0.5).astype(int)
    cat_metrics = {
        "accuracy": accuracy_score(y_test_sw, cat_preds),
        "precision": precision_score(y_test_sw, cat_preds, zero_division=0),
        "recall": recall_score(y_test_sw, cat_preds, zero_division=0),
        "f1_score": f1_score(y_test_sw, cat_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test_sw, cat_probs),
    }
    logging.info(f"CatBoost (threshold=0.5) => {cat_metrics}")

    logging.info("Done with chunk-based sliding windows + SMOTEENN approach for forecasting!")
    sys.exit(0)

if __name__ == "__main__":
    main()
