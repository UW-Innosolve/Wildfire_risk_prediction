import earthaccess
import pandas as pd
import h5py
import logging
import xarray as xr
import os
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# # Set up the Earthdata login credentials
# EARTHDATA_USERNAME = 'jromero7'
# EARTHDATA_PASSWORD = 'InnoSolve@UW7'

# Initialize the Earthdata login
earthaccess.login()


# Sample search for the ATL06 dataset
logger.info("Searching for the GLAH06 dataset in NSIDC DAAC")
results = earthaccess.search_data(
    short_name='GLAH06', # GLAS/ICESat L2 Global Land Surface Altimetry Data
    daac='NSIDC', # National Snow and Ice Data Center (NSIDC) DAAC
    bounding_box=(-10, 20, 10, 50),
    temporal=("2006-01", "2006-02"),
    count=1,
)

# Download the data
logger.info("Downloading the GLAH06 dataset")
output_dir = "scripts/data_collection/earthaccess_samples"
os.makedirs(output_dir, exist_ok=True)
logger.info("Downloading the GLAH06 dataset to {output_dir}...")
downloaded_files = earthaccess.download(results, output_dir)

# Replace with your downloaded file's path
h5_file_path = "/Users/jromero/Documents/GitHub/Wildfire_risk_prediction/scripts/data_collection/earthaccess_samples/GLAH06_634_2115_001_1288_1_01_0001.H5"

with h5py.File(h5_file_path, "r") as h5_file:
    # List all groups
    print("Groups in the file:", list(h5_file.keys()))

csv_output_dir = "csv_exports"
os.makedirs(csv_output_dir, exist_ok=True)


def convert_h5_to_csv(h5_file_path, output_dir):
    with h5py.File(h5_file_path, 'r') as h5_file:
        def extract_dataset(group, path=""):
            for key in group.keys():
                item = group[key]
                full_path = f"{path}/{key}".lstrip('/')
                if isinstance(item, h5py.Dataset):
                    # Convert dataset to a pandas DataFrame and save to CSV
                    data = item[:]
                    if data.ndim == 1:
                        df = pd.DataFrame(data, columns=[key])
                    else:
                        # Handle multidimensional datasets
                        df = pd.DataFrame(data)
                    csv_file = os.path.join(output_dir, f"{full_path.replace('/', '_')}.csv")
                    print(f"Exporting dataset {full_path} to {csv_file}")
                    df.to_csv(csv_file, index=False)
                    save_path = os.path.join(csv_output_dir, csv_file)
                    print(f"Dataset saved to {save_path}")
                elif isinstance(item, h5py.Group):
                    extract_dataset(item, path=full_path)

        print(f"Processing file: {h5_file_path}")
        extract_dataset(h5_file)

for h5_file_path in downloaded_files:
    logger.info(f"Converting {h5_file_path} to CSV")
    convert_h5_to_csv(h5_file_path, csv_output_dir)

print(f"CSV files have been saved to {csv_output_dir}")


# logger.info("Saving the GLAH06 dataset as a CSV file")
# df.to_csv("scripts/data_collection/earthaccess_samples/GLAH06_data_sample.csv")






files = earthaccess.download(results, "scripts/data_collection/earthaccess_samples")

1