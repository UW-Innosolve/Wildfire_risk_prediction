#!/usr/bin/env python3
import logging
import os
import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd

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
    Create training instances for forecasting. Each sample:
      - Input features: the flattened data of the previous 'lag_window' days in X
      - Target: aggregated over the *next* 'forecast_window' days in y (e.g. np.max for classification)

    X : Pandas DataFrame (indexed in time order)
    y : Pandas Series or 1D DataFrame aligned with X
    lag_window : int, how many past days to use as input
    forecast_window : int, how many future days to aggregate into target
    agg_func : function to aggregate the forecast period (e.g., np.max => if any day in next window is 1 => target=1)

    Returns:
      X_new : np.array of shape (n_samples, lag_window * n_features)
      y_new : np.array of shape (n_samples,)
    """
    n_samples = len(y) - lag_window - forecast_window + 1
    if n_samples <= 0:
        raise ValueError("Not enough data to create the specified sliding windows.")

    X_windows = []
    y_windows = []
    for i in range(n_samples):
        # Flatten the past 'lag_window' days of X into one row
        window_feats = X.iloc[i : i + lag_window].values.flatten()
        # Aggregate the next 'forecast_window' days of y
        target_val = agg_func(y.iloc[i + lag_window : i + lag_window + forecast_window].values)
        X_windows.append(window_feats)
        y_windows.append(target_val)

    return np.array(X_windows), np.array(y_windows)


def main():
    logging.info("Starting Forecasting Model (14-day lag, 3-day forecast) using XGBoost & CatBoost")

    # -------------------------------------------------------------------------
    # 1) Data Loading
    # -------------------------------------------------------------------------
    data_dir = "Scripts/data_processing/processed_data/split_data_dir"
    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")

    logging.info(f"Loading split data from: {data_dir}")
    
    # Using Dask to read CSVs, then converting to Pandas
    def load_dask_csv(path):
        import dask.dataframe as dd
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            sys.exit(1)
        logging.info(f"Reading: {path}")
        df_dd = dd.read_csv(path, assume_missing=True)
        df = df_dd.compute()
        logging.info(f"Read shape {df.shape} from {path}")
        return df

    X_train_df = load_dask_csv(X_train_path)
    X_test_df  = load_dask_csv(X_test_path)
    y_train_df = load_dask_csv(y_train_path)
    y_test_df  = load_dask_csv(y_test_path)

    # If y has shape (n, 1), flatten to 1D
    if y_train_df.shape[1] == 1:
        y_train_df = y_train_df.iloc[:, 0]
    if y_test_df.shape[1] == 1:
        y_test_df = y_test_df.iloc[:, 0]

    # Drop 'date' column if present
    if 'date' in X_train_df.columns:
        logging.info("Dropping 'date' column from X_train...")
        X_train_df.drop(columns=['date'], inplace=True)
    if 'date' in X_test_df.columns:
        logging.info("Dropping 'date' column from X_test...")
        X_test_df.drop(columns=['date'], inplace=True)

    logging.info(f"Final shapes -> X_train={X_train_df.shape}, y_train={y_train_df.shape}, "
                 f"X_test={X_test_df.shape}, y_test={y_test_df.shape}")

    # -------------------------------------------------------------------------
    # 2) Sliding Windows
    # -------------------------------------------------------------------------
    lag_window = 14
    forecast_window = 3
    agg_func = np.max  # if any day in the next 3 is a fire, target=1, else=0

    logging.info(f"Creating sliding windows with lag={lag_window}, forecast={forecast_window} using agg={agg_func.__name__}...")
    X_train_sw, y_train_sw = create_sliding_windows(X_train_df, y_train_df, lag_window, forecast_window, agg_func)
    X_test_sw,  y_test_sw  = create_sliding_windows(X_test_df,  y_test_df,  lag_window, forecast_window, agg_func)

    logging.info(f"Sliding window shapes -> X_train_sw={X_train_sw.shape}, y_train_sw={y_train_sw.shape}, "
                 f"X_test_sw={X_test_sw.shape}, y_test_sw={y_test_sw.shape}")

    # -------------------------------------------------------------------------
    # 3) GPU-Enabled XGBoost
    # -------------------------------------------------------------------------
    # (Cross-validation commented out)
    logging.info("Training XGBoost on sliding-window data (GPU if available)...")
    # For XGBoost 2.0+, recommended usage: 'tree_method' = 'hist', 'device' = 'cuda'
    xgb_clf = xgb.XGBClassifier(
        tree_method='hist',
        device='cuda',    # or 'cuda:0'
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    # fit
    xgb_clf.fit(X_train_sw, y_train_sw)

    # Evaluate with a basic threshold = 0.5
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

    # Evaluate with threshold = 0.7
    xgb_probs_07 = xgb_probs_05
    xgb_preds_07 = (xgb_probs_05 >= 0.7).astype(int)
    xgb_07_metrics = {
        "accuracy": accuracy_score(y_test_sw, xgb_preds_07),
        "precision": precision_score(y_test_sw, xgb_preds_07, zero_division=0),
        "recall":    recall_score(y_test_sw, xgb_preds_07, zero_division=0),
        "f1_score":  f1_score(y_test_sw, xgb_preds_07, zero_division=0),
        "roc_auc":   roc_auc_score(y_test_sw, xgb_probs_07),
    }
    logging.info(f"XGBoost (threshold=0.7) => {xgb_07_metrics}")

    # Optional: calibrate XGBoost
    logging.info("Calibrating XGBoost with isotonic regression (threshold=0.7 demo)...")
    from sklearn.calibration import CalibratedClassifierCV
    xgb_cal = CalibratedClassifierCV(base_estimator=xgb_clf, cv=3, method='isotonic')
    xgb_cal.fit(X_train_sw, y_train_sw)
    xgb_cal_probs = xgb_cal.predict_proba(X_test_sw)[:, 1]
    xgb_cal_preds = (xgb_cal_probs >= 0.7).astype(int)
    xgb_cal_metrics = {
        "accuracy": accuracy_score(y_test_sw, xgb_cal_preds),
        "precision": precision_score(y_test_sw, xgb_cal_preds, zero_division=0),
        "recall":    recall_score(y_test_sw, xgb_cal_preds, zero_division=0),
        "f1_score":  f1_score(y_test_sw, xgb_cal_preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test_sw, xgb_cal_probs),
    }
    logging.info(f"XGBoost + calibration (threshold=0.7) => {xgb_cal_metrics}")

    # -------------------------------------------------------------------------
    # 4) GPU-Enabled CatBoost
    # -------------------------------------------------------------------------
    logging.info("Training CatBoost on sliding-window data (GPU if available)...")
    cat_clf = CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        task_type='GPU',
        devices='0',  # if you have a single GPU
        eval_metric='Logloss',
        verbose=False
    )
    cat_clf.fit(X_train_sw, y_train_sw)

    # Evaluate at threshold=0.5
    cat_probs_05 = cat_clf.predict_proba(X_test_sw)[:, 1]
    cat_preds_05 = (cat_probs_05 >= 0.5).astype(int)
    cat_05_metrics = {
        "accuracy": accuracy_score(y_test_sw, cat_preds_05),
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
        "recall":    recall_score(y_test_sw, cat_preds_07, zero_division=0),
        "f1_score":  f1_score(y_test_sw, cat_preds_07, zero_division=0),
        "roc_auc":   roc_auc_score(y_test_sw, cat_probs_05),
    }
    logging.info(f"CatBoost (threshold=0.7) => {cat_07_metrics}")

    # Optional: calibrate CatBoost with isotonic regression
    logging.info("Calibrating CatBoost with isotonic regression (threshold=0.7 demo)...")
    cat_cal = CalibratedClassifierCV(base_estimator=cat_clf, cv=3, method='isotonic')
    cat_cal.fit(X_train_sw, y_train_sw)
    cat_cal_probs = cat_cal.predict_proba(X_test_sw)[:, 1]
    cat_cal_preds = (cat_cal_probs >= 0.7).astype(int)
    cat_cal_metrics = {
        "accuracy": accuracy_score(y_test_sw, cat_cal_preds),
        "precision": precision_score(y_test_sw, cat_cal_preds, zero_division=0),
        "recall":    recall_score(y_test_sw, cat_cal_preds, zero_division=0),
        "f1_score":  f1_score(y_test_sw, cat_cal_preds, zero_division=0),
        "roc_auc":   roc_auc_score(y_test_sw, cat_cal_probs),
    }
    logging.info(f"CatBoost + calibration (threshold=0.7) => {cat_cal_metrics}")

    # -------------------------------------------------------------------------
    # (Optional) Cross-Validation is commented out for faster iteration
    # -------------------------------------------------------------------------
    """
    # If you later want cross-validation:
    # from sklearn.model_selection import cross_val_score
    # scores = cross_val_score(xgb_clf, X_train_sw, y_train_sw, cv=5, scoring='f1')
    # logging.info(f"XGBoost CV F1 => {scores.mean()}")
    """

    logging.info("Finished forecasting pipeline (XGBoost & CatBoost) with a 3-day horizon using 14-day lag.")
    logging.info("Exiting pipeline.")

if __name__ == "__main__":
    main()
