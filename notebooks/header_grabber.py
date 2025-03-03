import pandas as pd

# Replace this with your CSV file path
file_path = r"C:\Users\ibuaz\OneDrive\Desktop\inno\EDA\fb_raw_data_201402.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Print all headers (column names)
print("Headers:")
for col in df.columns:
    print(col)
