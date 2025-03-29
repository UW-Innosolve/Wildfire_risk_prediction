import os
import pandas as pd


print("Current working directory:", os.getcwd())

def check_data_integrity(data_file):
    """
    Checks shape, missing values, columns, and a quick peek of the data file.
    """
    if not os.path.exists(data_file):
        print(f"[ERROR] File not found: {data_file}")
        return

    print(f"\nLoading {data_file}...")
    df = pd.read_csv(data_file)
    # print(f"Data shape: {df.shape}")

    # # Quick look at missing data
    # missing_count = df.isna().sum().sum()
    # print(f"Total missing values: {missing_count}")

    # Show column names
    print(f"Columns: {list(df.columns)}")

    # # Optional: show first 5 rows
    # print("\nSample rows:")
    print(df.head())

    # # Optional: If you have a label column (e.g., 'target'), show distribution
    # label_column = 'target'  # Adjust to your actual label column name
    # if label_column in df.columns:
    #     print(f"\nValue counts for {label_column}:")
    #     print(df[label_column].value_counts())

if __name__ == "__main__":
    # Adjust path as needed:
    data_file = "Scripts/data_processing/processed_data/Processed_data.csv"
    
    check_data_integrity(data_file)
