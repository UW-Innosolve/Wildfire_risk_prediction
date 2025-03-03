# ## This file contains classes related to the loading and pre-processing of invariant data.

# ## Outline


# import logging
# from cds_pipeline import CdsPipeline
# import pandas as pd
# import xarray as xr
# import cdsapi
# import os
# import urllib.request as request
# import ssl
# import certifi

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # # Initialize CDS pipeline
# # key = '734d2638-ef39-4dc1-bc54-4842b788fff6'
# # c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=key)

# # # Define the URL from the download button
# # low_veg_cover_url = "https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation?preview=/140385202/155324132/geo_1279l4_0.1x0.1.grb"
# # # Define the file name (extract from the URL or use a fixed name)
# # filename = low_veg_cover_url.split("/")[-1]  # Extracts the file name from the URL (last part)

# # # Define the directory to save the file
# # output_dir = "scripts/data_collection/static_datasets"
# # os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# # # Full path to the file
# # file_path = os.path.join(output_dir, filename)

# # # Check if the file already exists
# # if not os.path.exists(file_path):
# #     print(f"File not found locally. Downloading: {filename}")
# #     ssl._create_default_https_context = ssl._create_unverified_context

# #     # # Use certifi for SSL verification TODO: Fix SSL verification issue.
# #     # context = ssl.create_default_context(cafile=certifi.where())

# #     file = request.urlretrieve(low_veg_cover_url, "file1.grib")
# #     print(type(file))

# #     # Save the file to disk
# #     with open(file_path, 'wb') as f:
# #         f.write(file[1])
# #     print(f"File downloaded and saved as: {file_path}")
# # else:
# #     print(f"File already exists: {file_path}")

# #     # response = requests.get(low_veg_cover_url, stream=True)
# #     # response.raise_for_status()  # Raise an exception for HTTP errors

# #     # try:
# #     #     # Save the file to disk
# #     #     with open(filename, "wb") as file:
# #     #         response.content
# #     # except requests.exceptions.RequestException as e:
# #     #     print(f"An error occurred: {e}")



# # # # Load the data
# # # data = xr.open_dataset('output_file.nc')
# # # print(data)

# class InvarPreporcessor:
#     def __init__(self):
#         pass

#     def load_invar_data(self, invar_data_path):
#         """
#         Load invariant data from csv file at path.
#         """
#         invar_data = pd.read_csv(invar_data_path)
#         return invar_data
    
#     def create_space_df(self, lat_range, long_range, grid_resolution):
#         """
#         Create a space-time dataframe for time-invariant data.
#         """
#         latitudes = list(range(lat_range[0], lat_range[1], grid_resolution))
#         longitudes = list(range(long_range[0], long_range[1], grid_resolution))
#         invar_data = pd.DataFrame()
#         for lat in latitudes:
#             for lon in longitudes:
#                 invar_data = invar_data.append({'latitude': lat, 'longitude': lon}, ignore_index=True)
#         return invar_data
    
#     def replicate_across_time(self, invar_data, start_date, end_date):
#         """
#         Replicate the invariant data across time.
#         """
#         pass