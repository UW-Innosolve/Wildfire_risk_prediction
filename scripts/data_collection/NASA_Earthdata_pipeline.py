import earthaccess
import pandas as pd
import h5py
import logging
import xarray as xr
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

### NasaEarthdataPipeline Overview
## Initialization and Authentication
## - initialize Earthdata login
## - set up Earthdata login credentials

## Search and Download Bulk Dataset (H5) from Earthdata API via earthdata_pull functions
## - checks for dataset in local files first
## - Sets all parameters for search
## - called for each daac to be used

## Complile subsets from Downloaded .H5 File into dataframe via earthdata_slice
## - can accept data from multiple daacs
## - One time range and freuency set for all data

## Save Dataframes
## - Optionally save dataframes as CSV files


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


    ## earthdata_pull_invar
    ## - Searches and downloads the dataset from the Earthdata API
    ## - Single spatial snapshot dataset NOTE: This spatial nature is unverified.
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
    

    ## earthdata_pull_var
    ## - Searches and downloads the a full dataset from the Earthdata API
    ## - Multiple spatial snapshots dataset accross desired time range
    def earthdata_pull_var(self, short_name, daac, doi, bounding_box, temporal, time_range):
        pass


    ## earthdata_slice_to_csv
    ## - Slices the dataset into a dataframe from hardcoded parameters
    ## - Converts the dataframe to a CSV file then saves it
    ## - Returns the dataframe
    ## - NOTE: This functino and the parameters used from the H5 file are hardcoded
    def earthdata_slice(self, output_dir, h5_file, csv=False):
        dataset = pd.DataFrame()

        h5py_file = h5py.File(h5_file, "r")
        level1 = h5py_file.keys()
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
        

    ## earthdata_save_to_h5
    ## - Downloads the dataset from the Earthdata API
    ## - Saves the dataset to the output directory
    def earthdata_save_to_h5(self, h5_file, output_dir):
        downloaded_files = earthaccess.download(h5_file, output_dir)
        logger.info(f"Downloaded {len(downloaded_files)} files")
        return downloaded_files

        
   