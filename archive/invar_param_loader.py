# import requests
# response = requests.get('https://n5eil01u.ecs.nsidc.org/')
# print(response.status_code)

import earthaccess
import pandas as pd
import h5py
import logging
import xarray as xr
import os
import requests


# Replace with your downloaded file's path
h5_file_path = "/Users/jromero/Documents/GitHub/Wildfire_risk_prediction/scripts/data_collection/earthaccess_samples/GLAH06_634_2115_001_1288_1_01_0001.H5"

with h5py.File(h5_file_path, "r") as h5_file:
    # List all groups
    print("Groups in the file:", list(h5_file.keys()))