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
earthaccess.download(results, "scripts/data_collection/earthaccess_samples")


# Replace with your downloaded file's path
h5_file_path = "/Users/jromero/Documents/GitHub/Wildfire_risk_prediction/scripts/data_collection/earthaccess_samples/GLAH06_634_2115_001_1288_1_01_0001.H5"

with h5py.File(h5_file_path, "r") as h5_file:
    # List all groups
    print("Groups in the file:", list(h5_file.keys()))
    



# logger.info("Saving the GLAH06 dataset as a CSV file")
# df.to_csv("scripts/data_collection/earthaccess_samples/GLAH06_data_sample.csv")






files = earthaccess.download(results, "scripts/data_collection/earthaccess_samples")

1