#!/usr/bin/env python3
import pandas as pd
from collections import Counter

def main():
    data_file = "scripts/data_processing/processed_data/split_data_dir/y_train.csv"
    print(f"Reading training labels from: {data_file}")

    # We'll read the CSV in chunks to handle large files efficiently.
    chunk_size = 100000  # adjust as needed
    counts = Counter()
    total = 0

    # Process file in chunks
    for chunk in pd.read_csv(data_file, chunksize=chunk_size):
        # If there's only one column, assume that's the label column.
        if chunk.shape[1] == 1:
            col = chunk.columns[0]
            chunk_counts = chunk[col].value_counts().to_dict()
        else:
            # Otherwise, assume the first column is the label.
            chunk_counts = chunk.iloc[:, 0].value_counts().to_dict()
        counts.update(chunk_counts)
        total += len(chunk)

    print("\nClass distribution in y_train:")
    for label, count in counts.items():
        percentage = (count / total) * 100
        print(f"  Class {label}: {count} instances ({percentage:.2f}%)")

if __name__ == "__main__":
    main()
