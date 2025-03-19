import os
import glob
import pandas as pd


# Paths for the two folders
folder1 = "raw_data_wout_cffdrs/lightning"
folder2 = "raw_data_wout_cffdrs/other"
output_folder = "raw_data_wout_cffdrs/raw_merged"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get sorted list of CSV files from both folders
files1 = sorted([f for f in os.listdir(folder1) if f.endswith(".csv")])
files2 = sorted([f for f in os.listdir(folder2) if f.endswith(".csv")])

# Loop through files that are present in both folders
for file1, file2 in zip(files1, files2):
    if file1 == file2:  # Check if filenames match
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)

        # Load CSVs into dataframes
        df1 = pd.read_csv(path1)
        print(df1.shape)
        df2 = pd.read_csv(path2)
        if 'lightning_count' in df2.columns or 'absv_strength_sum'in df2.columns or 'multiplicity_sum' in df2.columns:
        # if ['lightning_count',	'absv_strength_sum',	'multiplicity_sum'] in df2.columns:
            df2 = df2.drop(columns=['lightning_count', 'absv_strength_sum', 'multiplicity_sum'])

        # Check if the shapes are the same
        if df1.shape[0] == df2.shape[0]:
            # Check if first 3 columns are 'date', 'latitude', 'longitude'
            if list(df1.columns[:3]) == ['date', 'latitude', 'longitude'] and list(df2.columns[:3]) == ['date', 'latitude', 'longitude']:
                # Merge on the first three columns
                merged_df = pd.merge(df1, df2, on=['date', 'latitude', 'longitude'], how='inner')

                # Save merged CSV to output folder
                output_path = os.path.join(output_folder, file1)
                merged_df.to_csv(output_path, index=False)
                print(f"Merged and saved: {file1}")
            else:
                print(f"Skipping {file1} - Columns mismatch")
        else:
            print(f"Skipping {file1} - Shape mismatch")
    else:
        print(f"Skipping unmatched files: {file1} and {file2}")

print("Merging complete!")
