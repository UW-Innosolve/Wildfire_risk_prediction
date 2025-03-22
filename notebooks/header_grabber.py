import pandas as pd

# Replace this with your CSV file path
file_path = "scripts/data_processing/raw_data_dir/complete_raw_data.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Print all headers (column names)
print("Headers:")
for col in df.columns:
    print(col)
