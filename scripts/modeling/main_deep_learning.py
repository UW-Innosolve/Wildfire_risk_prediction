#!/usr/bin/env python3
import os
import sys
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -----------------------------
# 1) Focal Loss for Imbalance
# -----------------------------
class FocalLoss(nn.Module):
    """
    Focal loss for binary classification: 
      FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        pt = torch.sigmoid(logits)
        pt = pt * targets + (1 - pt) * (1 - targets)
        focal = self.alpha * (1 - pt).pow(self.gamma) * bce_loss
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

# -----------------------------
# 2) The CNN+LSTM Model
# -----------------------------
class CNNLSTMModel(nn.Module):
    def __init__(self, feature_dim, cnn_channels=32, lstm_hidden=64):
        """
        feature_dim: number of features in one day (including lat, lon, etc.)
        cnn_channels: number of channels for the 1D CNN
        lstm_hidden: hidden dimension for LSTM
        """
        super().__init__()
        # Suppose we do a 1D CNN across the time dimension (lag_window),
        # expecting input shape: (batch, 1, lag_window, feature_dim)
        # We'll reorder inside the dataset to (batch, 1, lag_window, feature_dim).
        # Then convolve across dimension=2 (the time dimension).
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn_channels, kernel_size=(3, feature_dim), stride=(1, feature_dim)),
            nn.ReLU(),
            # shape => (batch, cnn_channels, lag_window-2, 1) effectively
        )

        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, 1)  # final output => 1 logit for binary classification

    def forward(self, x):
        """
        x shape: (batch, lag_window, feature_dim)
        We'll unsqueeze to (batch, 1, lag_window, feature_dim) for CNN.
        Then squeeze/transpose to feed LSTM, shape => (batch, lag_window-2, cnn_channels).
        """
        bsz, lag, fdim = x.size()

        # CNN expects (batch, channels=1, lag, feature_dim)
        x = x.unsqueeze(1)  # => (batch, 1, lag, feature_dim)
        c = self.cnn(x)     # => (batch, cnn_channels, (lag-2), 1) if kernel_size=(3, feature_dim)
        # remove the last dim => (batch, cnn_channels, lag-2)
        c = c.squeeze(-1)
        # we want (batch, (lag-2), cnn_channels) for LSTM (batch_first=True)
        c = c.transpose(1, 2)  # => (batch, lag-2, cnn_channels)

        out, (h, c0) = self.lstm(c)   # out => (batch, lag-2, lstm_hidden)
        # take the final time step
        final = out[:, -1, :]        # => (batch, lstm_hidden)
        logits = self.fc(final)      # => (batch, 1)
        return logits.squeeze(-1)    # => (batch,)

# -----------------------------
# 3) Sliding Window + Forecast
# -----------------------------
def read_data_in_chunks(X_path, y_path, chunk_start, chunk_end, overlap, lag_window, forecast_window):
    """
    Reads partial data with Dask -> pandas, adds overlap, then yields a chunk of rows.
    We'll build sliding windows from that chunk. Return X_windows, y_windows
    aggregator = np.max => if any day in next forecast_window is 1 => label=1
    """
    # We'll do a basic approach: read the entire chunk with overlap in memory,
    # then build windows. 
    # The script still "chunks" the dataset, but if chunk_end-chunk_start is smaller,
    # hopefully it fits in memory.

    # read via dask
    import dask.dataframe as dd
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        logging.error("File not found in read_data_in_chunks.")
        sys.exit(1)

    # Dask can read partial CSV by specifying blocksize or so, but here we do a simpler approach:
    # We'll read entire CSV, then slice. A more advanced approach might do row-based partial reads.

    X_dd = dd.read_csv(X_path, assume_missing=True)
    Y_dd = dd.read_csv(y_path, assume_missing=True)
    # compute
    X_all = X_dd.compute()
    Y_all = Y_dd.compute()

    # In HPC, better might be to read entire CSV once for each chunk. This is simplified.
    # Now we do slicing
    N = len(X_all)
    extended_start = max(0, chunk_start - overlap)
    extended_end   = min(N, chunk_end + overlap)

    chunk_X = X_all.iloc[extended_start:extended_end].reset_index(drop=True)
    chunk_y = Y_all.iloc[extended_start:extended_end].reset_index(drop=True)

    # We'll create sliding windows, only keep windows that start in [chunk_start, chunk_end - lag - forecast]
    # aggregator => np.max
    # first define aggregator
    aggregator = np.max

    # Build windows
    local_len = len(chunk_X)
    X_list, y_list = [], []
    max_start = local_len - (lag_window + forecast_window)
    if max_start < 0:
        return np.empty((0, lag_window, chunk_X.shape[1])), np.empty((0,))

    for local_start in range(max_start + 1):
        global_start = extended_start + local_start
        # Keep if global_start in [chunk_start, chunk_end - lag - forecast]
        if global_start < chunk_start or global_start > (chunk_end - lag_window - forecast_window):
            continue

        # X slice => shape (lag_window, n_features)
        feats = chunk_X.iloc[local_start : local_start+lag_window].values
        label_slice = chunk_y.iloc[local_start+lag_window : local_start+lag_window+forecast_window].values
        label_val = aggregator(label_slice)
        X_list.append(feats)
        y_list.append(label_val)

    if not X_list:
        return np.empty((0, lag_window, chunk_X.shape[1])), np.empty((0,))
    return np.array(X_list), np.array(y_list)

# -----------------------------
# 4) A Minimal PyTorch Dataset
# -----------------------------
class ForecastDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------------
# 5) Main
# -----------------------------
def main():
    logging.info("Starting a chunk-based CNN+LSTM forecasting approach with focal loss for imbalance...")

    # Print columns + first 2 lines to stderr
    # We'll read just the columns from X_train
    # Then read the first 2 lines from the CSV
    # so that we can see them in .err file
    import sys
    X_head = pd.read_csv("scripts/data_processing/processed_data/split_data_dir/X_train.csv", nrows=2)
    sys.stderr.write(f"Columns: {X_head.columns.tolist()}\n")
    sys.stderr.write(f"First 2 lines of X_train:\n{X_head}\n")

    # Hyperparams
    lag_window = 14
    forecast_window = 3
    chunk_size = 500_000
    overlap = lag_window + forecast_window - 1

    # Where your CSVs are
    X_train_path = "scripts/data_processing/processed_data/split_data_dir/X_train.csv"
    y_train_path = "scripts/data_processing/processed_data/split_data_dir/y_train.csv"
    N_train = 5258440  # if you know the length in advance

    # We'll do an epoch-based approach: we pass through the entire dataset chunk by chunk once
    # Memory usage is constant because we load one chunk => build windows => train => free.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTMModel(
        feature_dim=74,  # after dropping 'date'
        cnn_channels=32,
        lstm_hidden=64
    ).to(device)

    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # One pass (1 epoch) for demonstration
    # For multiple epochs, you'd wrap the entire chunk loop in a for epoch in range(epochs): ...

    start_idx = 0
    while start_idx < N_train:
        chunk_end = min(start_idx + chunk_size, N_train)

        logging.info(f"Processing chunk in range [{start_idx}, {chunk_end}) with overlap...")
        X_chunk_win, y_chunk_win = read_data_in_chunks(
            X_train_path, y_train_path,
            chunk_start=start_idx,
            chunk_end=chunk_end,
            overlap=overlap,
            lag_window=lag_window,
            forecast_window=forecast_window
        )
        logging.info(f"Chunk windows shape => X={X_chunk_win.shape}, y={y_chunk_win.shape}")

        if X_chunk_win.shape[0] == 0:
            logging.info("No windows in this chunk. Moving on.")
            start_idx = chunk_end
            continue

        # We'll create a PyTorch Dataset+DataLoader
        # Then do a training loop (one pass)
        ds = ForecastDataset(X_chunk_win, y_chunk_win)
        loader = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)

        # Training loop (single pass)
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)  # shape => (batch, lag_window, feature_dim)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)  # => shape (batch,)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        logging.info(f"Done training on chunk [{start_idx}, {chunk_end}).")

        start_idx = chunk_end

    logging.info("Done with entire training pass on chunk-based approach.")
    # You might save your model:
    torch.save(model.state_dict(), "cnn_lstm_forecast.pth")

    # Evaluate on test
    # We'll do a simpler approach: read entire test in one pass if it's smaller
    # Build windows, do inference
    logging.info("Evaluating on test set...")

    # Build windows for entire test if feasible
    # If too big, chunk the test similarly
    # For demonstration, we do single pass
    from dask import dataframe as dd
    X_test_dd = dd.read_csv("scripts/data_processing/processed_data/split_data_dir/X_test.csv", assume_missing=True)
    Y_test_dd = dd.read_csv("scripts/data_processing/processed_data/split_data_dir/y_test.csv", assume_missing=True)
    X_test_all = X_test_dd.compute().drop(columns=['date'], errors='ignore')
    Y_test_all = Y_test_dd.compute().iloc[:, 0]

    # Build windows in memory
    # We'll do a direct approach
    def sliding_windows_test(Xdf, Ydf, lag, forecast, aggregator=np.max):
        N = len(Xdf)
        nfeat = Xdf.shape[1]
        max_start = N - (lag + forecast)
        if max_start < 0:
            return np.empty((0, lag, nfeat)), np.empty((0,))
        Xlist, Ylist = [], []
        for i in range(max_start + 1):
            feats = Xdf.iloc[i:i+lag].values
            label_vals = Ydf.iloc[i+lag : i+lag+forecast].values
            label = aggregator(label_vals)
            Xlist.append(feats)
            Ylist.append(label)
        return np.array(Xlist), np.array(Ylist)

    X_test_sw, y_test_sw = sliding_windows_test(X_test_all, Y_test_all, lag_window, forecast_window)
    logging.info(f"Test windows => X={X_test_sw.shape}, y={y_test_sw.shape}")

    # Inference
    model.eval()
    ds_test = ForecastDataset(X_test_sw, y_test_sw)
    loader_test = DataLoader(ds_test, batch_size=512, shuffle=False)
    preds, probs, tgts = [], [], []
    with torch.no_grad():
        for bx, by in loader_test:
            bx = bx.to(device)
            by = by.to(device)
            logits = model(bx)
            p = torch.sigmoid(logits)
            # threshold=0.5
            pred = (p >= 0.5).long()
            preds.append(pred.cpu().numpy())
            probs.append(p.cpu().numpy())
            tgts.append(by.cpu().numpy())

    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    tgts  = np.concatenate(tgts)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    accuracy = accuracy_score(tgts, preds)
    precision= precision_score(tgts, preds, zero_division=0)
    recall   = recall_score(tgts, preds, zero_division=0)
    f1       = f1_score(tgts, preds, zero_division=0)
    roc      = roc_auc_score(tgts, probs)
    logging.info(f"Test metrics => accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, roc_auc={roc:.4f}")

    logging.info("All done! Exiting script.")

if __name__ == "__main__":
    main()
