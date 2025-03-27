import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def optimize_dtypes(df):
    """
    Downcasts numeric columns to reduce memory usage.
    """
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def split_and_filter(
    data_file,
    output_dir,
    label_col="is_fire_day",
    test_size=0.2,
    random_state=42,
    chunksize=500000
):
    """
    1) Loads the dataset in chunks from `data_file`.
    2) Parses the 'date' column and keeps only rows where the month is in [3..11].
    3) Drops rows with ANY missing values.
    4) Downcasts numeric columns to optimize memory.
    5) Concatenates the processed chunks.
    6) Splits into train/test sets (80/20 by default).
    7) Saves the resulting CSV files to `output_dir`.
    """
    if not os.path.exists(data_file):
        print(f"[ERROR] File not found: {data_file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from {data_file} in chunks...")
    processed_chunks = []
    start_time = time.time()
    
    for i, chunk in enumerate(pd.read_csv(data_file, chunksize=chunksize, low_memory=False)):
        # Convert 'date' to datetime if it exists
        if "date" in chunk.columns:
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            # Filter to keep only rows in months 3..11
            month_mask = chunk["date"].dt.month.between(3, 11)
            chunk = chunk[month_mask]
        else:
            print(f"[ERROR] No 'date' column found in chunk {i}. Skipping chunk.")
            continue

        # Drop rows with any missing values
        chunk.dropna(axis=0, how="any", inplace=True)
        # Optimize numeric data types
        chunk = optimize_dtypes(chunk)
        processed_chunks.append(chunk)
        print(f"Processed chunk {i+1} with shape: {chunk.shape}")
    
    df = pd.concat(processed_chunks, ignore_index=True)
    elapsed = time.time() - start_time
    print(f"After processing, dataset shape: {df.shape} (Time: {elapsed:.2f} seconds)")

    if label_col not in df.columns:
        print(f"[ERROR] Label column '{label_col}' not found in dataset.")
        return

    # Separate features (X) and target (y)
    X = df.drop(columns=[label_col])
    y = df[label_col]

    # Train/test split (stratify if needed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )

    # Save the CSV files
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"\nSuccessfully created train/test splits:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    data_file = "Scripts/data_processing/processed_data/processed_data.csv"
    output_dir = "Scripts/data_processing/processed_data/split_data_dir"
    
    split_and_filter(
        data_file=data_file,
        output_dir=output_dir,
        label_col="is_fire_day",
        test_size=0.2,
        random_state=42
    )
