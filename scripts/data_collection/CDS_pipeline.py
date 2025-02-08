import cdsapi
import pandas as pd
import numpy as np
import xarray as xr  # Import xarray for working with GRIB files
from datetime import timedelta
import requests
import tempfile
import os.path
import logging
import sys
sys.path.append('scripts/utils/authentication')
from firebird_auth import FirebirdAuth ## TODO: Optimize modular organization for the project.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CdsPipeline:
    def __init__(self, key):
        self.var_variables = []
        self.invar_variables = []
        self.cds_request_parameters = {}
        #self.cds_api_key = ''
        self.CDS_client = None
        
        ## Obtain key from from credentials file
        self.cds_api_key = key #FirebirdAuth().get_cds_key("scripts/utils/authentication/credentials.json")
        ## Initialize CDS client
        self.CDS_client = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=self.cds_api_key)
        logger.info("""CDS Pipeline (client) has been initialized.
                    The following methods must be called before an API call can be made:
                        - set_variant_variables(self, var_variables)
                        - set_invariant_variables(self, invar_variables)
                        - set_request_parameters(self, var_variables, invar_variables, lat_range, long_range, grid_resolution)""")


    def set_variant_variables(self, var_variables):
        """Set the time-variant variables for the CDS API request"""
        self.var_variables = var_variables
        logger.info(f"Time-variant variables set: {self.var_variables}")

    ## set_invariant_variables method
    ##          - set the time-invariant variables for the CDS API request
    ##          - must be called before fetch_weather_data method
    def set_invariant_variables(self, invar_variables):
        """Set the time-invariant variables for the CDS API request"""
        self.invar_variables = invar_variables
        logger.info(f"Time-invariant variables set: {self.invar_variables}")

    ## set_request_parameters method
    ##          - set the parameters for the CDS API request including latitude range, longitude range, and grid resolution
    ##          - time-variant variables and time-invariant variables lists must be set before calling this method and cannot be empty
    ##          - must be called before fetch_weather_data method
    ##          - input parameters: var_variables, invar_variables, lat_range, long_range, grid_resolution
    ##          - mutates self.cds_request_parameters
    def set_request_parameters(self, var_variables, invar_variables, lat_range, long_range, grid_resolution):
        """Set the parameters for the CDS API request including time-variant variables, time-invariant variables, latitude range, longitude range, and grid resolution
            input parameters:
                - var_variables: list of time-variant variables from the CDS API
                - invar_variables: list of time-invariant variables from the CDS API
                - lat_range: latitude range for the data request as a list [min_lat, max_lat]
                - long_range: longitude range for the data request as a list [min_long, max_long]
                - grid_resolution: resolution of the grid for the data request, in degrees"""
        variables = list(set(self.var_variables + self.invar_variables))
        
        self.cds_request_parameters = {
            'format': 'grib',
            'variable': variables,
            'area': [lat_range[1], long_range[0], lat_range[0], long_range[1]],  # [north, west, south, east]
            'grid': [grid_resolution, grid_resolution]
        }
        logger.info(f"Request parameters set: {self.cds_request_parameters}")

    ## _read_grib_to_dataframe method
    ##          - read the GRIB file into a DataFrame
    ##          - input: grib_file
    ##          - output: df (pandas DataFrame)
    ##          - private method
    def _read_grib_to_dataframe(self, grib_file):
        """Read the GRIB file into a DataFrame with enhanced error handling.
        
        If the file is very small (indicating an error page or incomplete file) or is a ZIP archive,
        this function logs an error or attempts to unzip it before parsing.
        """
        import zipfile

        try:
            # Log the downloaded file size for debugging.
            file_size = os.path.getsize(grib_file)
            logger.info(f"Downloaded file size: {file_size} bytes")
            if file_size < 10000:  # adjust threshold as needed; 10KB is an example threshold
                logger.error(f"File size {file_size} bytes is too small; likely not a valid GRIB file.")
                raise ValueError("Downloaded file is too small, may be an error page or truncated file.")
            
            # If the file is actually a zip archive, unzip it first.
            if zipfile.is_zipfile(grib_file):
                logger.info("Downloaded file is a ZIP archive. Unzipping...")
                with zipfile.ZipFile(grib_file, 'r') as z:
                    # Assume the ZIP contains one GRIB file; take the first.
                    grib_names = z.namelist()
                    if not grib_names:
                        raise ValueError("ZIP archive is empty.")
                    # Extract the first file to a temporary location.
                    extracted_file = os.path.join(os.path.dirname(grib_file), grib_names[0])
                    z.extract(grib_names[0], os.path.dirname(grib_file))
                    logger.info(f"Extracted {grib_names[0]} from ZIP archive.")
                    # Attempt to read the extracted GRIB file.
                    ds = xr.open_dataset(extracted_file, engine='cfgrib')
                    df = ds.to_dataframe().reset_index()
                    df['date'] = pd.to_datetime(df['time']).dt.normalize()
                    df = df.drop(columns=['number'], errors='ignore')
                    logger.info(f"GRIB file '{extracted_file}' successfully read into DataFrame.")
                    os.remove(extracted_file)  # Cleanup the extracted file.
                    return df
            else:
                # If not a ZIP, try reading the file directly.
                ds = xr.open_dataset(grib_file, engine='cfgrib')
                df = ds.to_dataframe().reset_index()
                df['date'] = pd.to_datetime(df['time']).dt.normalize()
                df = df.drop(columns=['number'], errors='ignore')
                logger.info(f"GRIB file '{grib_file}' successfully read into DataFrame.")
                return df

        except Exception as e:
            logger.error(f"Error reading GRIB file '{grib_file}': {e}")
            return None


    ## fetch_weather_data method
    ##          - fetch weather data from the CDS API using the specified request parameters
    ##          - invariant variables must be set before calling this method, method cannot be called without at least one invariant variable set
    ##          - variant variables must be set before calling this method, method cannot be called without at least one variant variable set
    def fetch_weather_data(self, start_date, end_date):
        """Fetch weather data from the CDS API using the specified request parameters.
        
        This function downloads the data to a temporary file, logs the file size for debugging,
        and then attempts to read the file using _read_grib_to_dataframe. It cleans up the temporary
        file afterward.
        """
        # Ensure all required parameters have been set
        if not self.cds_request_parameters:
            raise ValueError("Request parameters have not been set. Please call set_request_parameters first.")
        elif not self.var_variables:
            raise ValueError("Time-variant variables have not been set. Please call set_variant_variables first.")
        elif not self.invar_variables and any(var in self.cds_request_parameters['variable'] for var in self.invar_variables):
            raise ValueError("Time-invariant variables have not been set. Please call set_invariant_variables first.")

        try:
            # Set dates for the request
            dates = {
                'year': list(set([str(date.year) for date in pd.date_range(start=start_date, end=end_date)])),
                'month': list(set([str(date.month).zfill(2) for date in pd.date_range(start=start_date, end=end_date)])),
                'day': list(set([str(date.day).zfill(2) for date in pd.date_range(start=start_date, end=end_date)])),
                'time': ['12:00']
            }

            # Merge cds_request_parameters and dates
            request_parameters = {**self.cds_request_parameters, **dates}
            logger.info(f"Fetching weather data with parameters: {request_parameters}")

            # Set up temporary file to store the GRIB data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".grib") as tmp_file:
                target_file = tmp_file.name

            # Make the API call to retrieve the data and store it in the temporary file
            self.CDS_client.retrieve('reanalysis-era5-land', request_parameters, target_file)
            logger.info(f"Weather data successfully retrieved and saved to '{target_file}'.")

            # Log the downloaded file size
            file_size = os.path.getsize(target_file)
            logger.info(f"Downloaded GRIB file size: {file_size} bytes")

            # Read the GRIB file using the updated function
            df = self._read_grib_to_dataframe(target_file)
            if df is None:
                logger.error("Failed to parse the GRIB file into a DataFrame.")
                return None

            # Filter weather data to ensure it's within the correct date range
            df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]

            # Convert weather_df 'date' to datetime.date type for matching purposes
            df['date'] = df['date'].dt.date

            os.remove(target_file)
            logger.info(f"Temporary GRIB file '{target_file}' has been removed.")

            return df

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error during data retrieval: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            if 'target_file' in locals() and os.path.exists(target_file):
                os.remove(target_file)
                logger.info(f"Temporary GRIB file '{target_file}' has been removed due to an error.")
            return None
