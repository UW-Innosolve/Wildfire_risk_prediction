import earthaccess
import pandas as pd
import h5py
import logging
import xarray as xr
import os
import requests
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# # Set up the Earthdata login credentials
# EARTHDATA_USERNAME = 'jromero7'
# EARTHDATA_PASSWORD = 'InnoSolve@UW7'




### NasaEarthdataPipeline Overview
## Initialization and Authentication
## - initialize Earthdata login
## - set up Earthdata login credentials

## Search and Download Bulk Dataset (H5)
## - checks for dataset in local files first
## - Sets all parameters for search
## - called for each daac to be used

## Complile subsets from Downloaded .H5 File into dataframe
## - can accept data from multiple daacs
## - One time range and freuency set for all data

## Save Dataframe as CSV
## - saves the compiled dataframe as a CSV file


class NasaEarthdataPipeline:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        os.environ['EARTHDATA_USERNAME'] = self.username
        os.environ['EARTHDATA_PASSWORD'] = self.password
        try:
            self.auth = earthaccess.login()
            logger.info("Earthdata login  successful")
        except Exception as e:
            logger.error(f"Earthdata login failed: {e}")
            raise e
        

    def initialize_earthdata(self):
        pass


    ## earthdata_pull
    ## - Searches and downloads the dataset from the Earthdata API
    ## - single spatial snapshot dataset
    def earthdata_pull_invar(self, short_name, daac, doi, bounding_box, temporal):
        ## Create search query
        h5_file = earthaccess.search_data(
            short_name=short_name,
            daac=daac,
            doi = doi,
            bounding_box=bounding_box,
            temporal=temporal,
            count=1, ## Only one file since it is being used as a snapshot in invariant data
        )

        logger.info(f"Searching for the {short_name} dataset in {daac} DAAC")
        logger.info(f"Downloading {short_name}_{daac} h5 file")
        downloaded_files = self.earthdata_save_to_h5(h5_file, "scripts/data_collection/static_datasets")
        return downloaded_files
    

    ## earthdata_slice_to_csv
    ## - Slices the dataset into a dataframe from hardcoded parameters
    ## - Converts the dataframe to a CSV file then saves it
    ## - Returns the dataframe

    def earthdata_slice(self, output_dir, h5_file, csv=False):
        dataset = pd.DataFrame()

        h5py_file = h5py.File(h5_file, "r")
        level1 = h5py_file.keys()
        # level2 = h5py_file[level1[0]].keys()
        print(level1)

        g1 = h5py_file.get('Data_40HZ')

        ## Datasets from the selected groups
        g1_1 = g1.get('Elevation_Surfaces')
        g1_1_elev = g1_1.get('d_elev') # Elevation
        logger.debug(f"d_elev: {g1_1_elev}")

        g1_2 = g1.get('Geolocation')
        g1_2_lat = g1_2.get('d_lat') # Latitude
        g1_2_lon = g1_2.get('d_lon') # Longitude
        logger.debug(f"d_lat: {g1_2_lat}")
        logger.debug(f"d_lon: {g1_2_lon}")

        g1_3 = g1.get('Time')
        g1_3_time = g1_3.get('d_UTCTime_40') # Time of measurement
        logger.debug(f"d_UTCTime_40: {g1_3_time}")

        ## Convert to numpy arrays
        times = np.array(g1_3_time[:])
        lats = np.array(g1_2_lat[:])
        lons = np.array(g1_2_lon[:])
        elevs = np.array(g1_1_elev[:])

        ## Create dataframe
        dataset = pd.DataFrame({
            'Time': times,
            'Latitude': lats,
            'Longitude': lons,
            'Elevation': elevs
        })
        logger.debug(f"Sliced dataset: {dataset}")

        ## Save as CSV if csv is True
        if csv:
            output_file_path = os.path.join(output_dir, "earthdata.csv")
            dataset.to_csv(output_file_path, index=False)
            logger.info(f"Dataframe saved as CSV to {output_dir}")

        return dataset
        

    def earthdata_save_to_h5(self, h5_file, output_dir):
        downloaded_files = earthaccess.download(h5_file, output_dir)
        logger.info(f"Downloaded {len(downloaded_files)} files")
        return downloaded_files

        
   

# def convert_h5_to_csv(h5_file_path, output_dir):
#     with h5py.File(h5_file_path, 'r') as h5_file:
#         def extract_dataset(group, path=""):
#             for key in group.keys():
#                 item = group[key]
#                 full_path = f"{path}/{key}".lstrip('/')
#                 if isinstance(item, h5py.Dataset):
#                     # Convert dataset to a pandas DataFrame and save to CSV
#                     data = item[:]
#                     if data.ndim == 1:
#                         df = pd.DataFrame(data, columns=[key])
#                     else:
#                         # Handle multidimensional datasets
#                         df = pd.DataFrame(data)
#                     csv_file = os.path.join(output_dir, f"{full_path.replace('/', '_')}.csv")
#                     print(f"Exporting dataset {full_path} to {csv_file}")
#                     df.to_csv(csv_file, index=False)
#                     save_path = os.path.join(csv_output_dir, csv_file)
#                     print(f"Dataset saved to {save_path}")
#                 elif isinstance(item, h5py.Group):
#                     extract_dataset(item, path=full_path)

#         print(f"Processing file: {h5_file_path}")
#         extract_dataset(h5_file)

# for h5_file_path in downloaded_files:
#     logger.info(f"Converting {h5_file_path} to CSV")
#     convert_h5_to_csv(h5_file_path, csv_output_dir)

# print(f"CSV files have been saved to {csv_output_dir}")


# # logger.info("Saving the GLAH06 dataset as a CSV file")
# # df.to_csv("scripts/data_collection/earthaccess_samples/GLAH06_data_sample.csv")






# files = earthaccess.download(results, "scripts/data_collection/earthaccess_samples")
