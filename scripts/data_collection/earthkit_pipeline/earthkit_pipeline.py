import pandas as pd
import numpy as np
import xarray as xr  # Import xarray for working with GRIB files
import cfgrib
import requests
import urllib.request as request # Import request for specifically for downloading files (invar files downloaded from URLs)
import tempfile
import os.path
import logging
import ssl
import zipfile
import earthkit.data


# from collection_utils.raw_data_assembly import RawDataAssembler


## CDS_pipeline class
## Should be initialized with a CDS API key
## Innosolve key: '734d2638-ef39-4dc1-bc54-4842b788fff6'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EkPipeline:
    def __init__(self, key):
        # Set the CDS API key in the environment
        self.key = key
        os.environ['CDSAPI_KEY'] = self.key
        
        self.cds_time_var_params = []
        self.cds_time_invar_params = []
        self.cds_accum_params = [] ## NOTE: Accumulation parameters must be handled separately.
        
        self.ek_req_call_1 = {} ## First earthkit call (time-variant and time-invariant parameters)
        self.ek_req_call_2 = {} ## Second earthkit call (accumulation parameters)
        
        self.ek_dataset = None
        
        # Set the CDS API key in the environment
        os.environ['CDSAPI_KEY'] = self.key
        logger.info("""Earthkit Pipeline has been initialized.
                    The following methods must be called before an API call can be made:
                        - set_variant_variables(self, var_variables)
                        - set_invariant_variables(self, invar_variables)
                        - set_request_parameters(self, var_variables, invar_variables, lat_range, long_range, grid_resolution)""")


    ## set_var_params method
    ##          - set the time-variant variables for the CDS API request
    ##          - must be called before fetch_weather_data method
    ##          - input parameters: var_params (list of time-variant variables from the CDS API)
    ##          - mutates self.var_variables
    def _set_var_params(self, var_params):
        """Set the time-variant variables for the CDS era5 request"""
        self.cds_time_var_params = var_params
        logger.info(f"Time-variant variables set: {self.cds_time_var_params}")
    
    
    ## set_invariant_variables method
    ##          - set the time-invariant variables for the CDS API request
    ##          - must be called before fetch_weather_data method
    ##          - input parameters: invar_variables (list of time-invariant parameters from url list below)
    ##          - private method
    def _set_invar_params(self, invar_params):
        """Set the time-invariant variables for the CDS era5 request"""
        self.cds_time_invar_params = invar_params
        logger.info(f"Time-invariant variables set: {self.cds_time_invar_params}")
    
    
    ## set_accum_params method
    ##          - set the accumulation variables for the CDS API request
    ##          - must be called before fetch_weather_data method
    ##          - input parameters: accum_params (list of accumulation variables from the CDS API)
    ##          - private method
    def _set_accum_params(self, accum_params):
        """Set the accumulation variables for the CDS era5 request"""
        self.cds_accum_params = accum_params
        logger.info(f"Accumulation variables set: {self.cds_accum_params}")
        

    ## set_request_parameters method
    ##          - set the parameters for the CDS API request including latitude range, longitude range, and grid resolution
    ##          - time-variant variables and time-invariant variables lists must be set before calling this method and cannot be empty
    ##          - must be called before fetch_weather_data method
    ##          - input parameters: var_variables, invar_variables, lat_range, long_range, grid_resolution
    ##          - mutates self.ek_request_parameters
    ##          - mutates self.var_params and self.invar_params
    def set_cds_request_parameters(self, var_params, invar_params, accum_params, lat_range, long_range, grid_resolution):
        """Set the parameters for the CDS API request including time-variant variables, time-invariant variables, latitude range, longitude range, and grid resolution
            input parameters:
                - var_variables: list of time-variant variables from the CDS API
                - invar_variables: list of time-invariant variables from the CDS API
                - lat_range: latitude range for the data request as a list [min_lat, max_lat]
                - long_range: longitude range for the data request as a list [min_long, max_long]
                - grid_resolution: resolution of the grid for the data request, in degrees"""
    
        self._set_var_params(var_params)
        self._set_invar_params(invar_params)
        self._set_accum_params(accum_params)
        
        ## Time variant and invariant parameters can be combined into a single list
        ##  - Earthkit is capable of handling both variant and invariant parameters together in on call
        ##  - Accumulation parameters must be handled separately
        var_and_invar_params_list = self.cds_time_var_params + self.cds_time_invar_params
        accum_params_list = self.cds_accum_params
        
        ## Non-temporal request parameters for the first earthkit call
        ## First call will be for the variant and invariant parameters
        var_and_invar_param_req_dict = dict(
            variable = var_and_invar_params_list,
            product_type = "reanalysis",
            area = [lat_range[1], long_range[0], lat_range[0], long_range[1]],  # [north, west, south, east]
            grid = [grid_resolution, grid_resolution],
            time = ['12:00']
            ## NOTE: Dates and times are set in the fetch_var_data method
        )
        
        ## Non-temporal request parameters for the second earthkit call
        ## Second call will be for the accumulation parameters
        accum_params_req_dict = dict(
            variable = accum_params_list, ## NOTE: Earthkit is capable of handling variant and invariant parameter requests
            product_type = "reanalysis",
            area = [lat_range[1], long_range[0], lat_range[0], long_range[1]],  # [north, west, south, east]
            grid = [grid_resolution, grid_resolution]
            # NOTE: No explicit time parameter is set for accumulation parameters
        )
        
        self.ek_req_call_1 = var_and_invar_param_req_dict
        self.ek_req_call_2 = accum_params_req_dict  
        
        
    ## _grib_to_df method
    ##          - read the GRIB file into a DataFrame, handles archive files and small files
    ##          - input: grib_file
    ##          - output: df (pandas DataFrame)
    ##          - private method
    def _grib_to_df(self, grib_file):
        """Read the GRIB file into a DataFrame with enhanced error handling.
        
        If the file is very small (indicating an error page or incomplete file) or is a ZIP archive,
        this function logs an error or attempts to unzip it before parsing.
        """
        
        file = grib_file
        logger.info(f"Reading GRIB file '{file}' into DataFrame...")

        try:
            # Log the downloaded file size for debugging.
            file_size = os.path.getsize(grib_file)
            if file_size < 5000:  #
                logger.error(f"File size {file_size} bytes is too small (<5kB); likely not a valid GRIB file.")
                raise ValueError("Downloaded file is too small, may be an error page or truncated file.")
            
            # If the file is actually a zip archive, unzip it first.
            if zipfile.is_zipfile(grib_file):
                logger.info("Downloaded file is a ZIP archive. Unzipping...")
                with zipfile.ZipFile(grib_file, 'r') as z:
                    # Assume the ZIP contains one GRIB file; take the first.
                    grib_names = z.namelist()
                    if len(grib_names) > 1:
                        logger.warning(f"ZIP archive contains multiple files: {grib_names}. Using the first.")
                    if not grib_names:
                        raise ValueError("ZIP archive is empty.")
                    # Extract the first file to a temporary location.
                    extracted_file = os.path.join(os.path.dirname(grib_file), grib_names[0])
                    z.extract(grib_names[0], os.path.dirname(grib_file))
                    logger.info(f"Extracted {grib_names[0]} from ZIP archive.")
                    file = extracted_file
            
            # Then read the file directly.
            ds = xr.open_dataset(file, engine= "earthkit")
            df = ds.to_dataframe().reset_index()
            logger.info(f"GRIB file '{file}' successfully read into DataFrame.")
            return df
        
        except Exception as e:
            logger.error(f"Error reading GRIB file '{grib_file}': {e}")
            return None


    ## ek_fetch_data method
    ##          - fetch time-variant and time-invariant weather data from the ERA5 dataset using the CDS API (via Earthkit)
    ##          - invariant variables must be set before calling this method, method cannot be called without at least one invariant variable set
    ##          - variant variables must be set before calling this method, method cannot be called without at least one variant variable set
    def ek_fetch_data(self, batch_dates):
        """Fetch weather data from the CDS API using the specified request parameters.
        
        This function downloads the data to a temporary file, logs the file size for debugging,
        and then attempts to read the file using _grib_to_df. It cleans up the temporary
        file afterward.
        """
        # Ensure all required parameters have been set
        if not self.cds_time_var_params or not self.cds_time_invar_params or not self.cds_accum_params:
            raise ValueError("Request parameters (variant, invariant, and accumulated) have not been set. Please call set_request_parameters first.")

        try:
            ## Convert batch_dates to list of date strings
            batch_timestamps = pd.to_datetime(batch_dates)
            batch_dates_only_str = pd.DataFrame({'date': batch_timestamps.astype(str)})  # Convert to string if needed
            batch_dates_only_list = batch_dates_only_str['date'].tolist()
            
            dates_dict = dict(
                date = batch_dates_only_list,
                )

            # Merge cds_request_parameters and dates dictionaries
            call_1_params = {**self.ek_req_call_1, **dates_dict}
            call_2_params = {**self.ek_req_call_2, **dates_dict}
            
            # Create two separate temp files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".grib") as tmp1, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".grib") as tmp2:
                call_1_file = tmp1.name
                call_2_file = tmp2.name 
                
            
            # Make the CDS API call 1 for time-variant and time-invariant parameters
            logger.info(f"Making CDS API call 1 for time-variant and time-invariant parameters: {call_1_params}")
            call_1_ds = earthkit.data.from_source(
                "cds",
                "reanalysis-era5-single-levels",
                call_1_params,
            )
            if call_1_ds is None:
                logger.error("CDS call 1 dataset is empty. No data was retrieved.")
                return None
            call_1_ds.save(call_1_file)
            file_1_size = os.path.getsize(call_1_file)
            logger.info(f"""Weather data retrieved and saved to '{call_1_file}',
                            File type {type(call_1_file)} and size {file_1_size} bytes""")
            
            # Make the CDS API call 2 for accumulation parameters
            logger.info(f"Making CDS API call 2 for accumulation parameters: {call_2_params}")
            call_2_ds = earthkit.data.from_source(
                "cds",
                "reanalysis-era5-single-levels",
                call_2_params
            )
            if call_2_ds is None:
                logger.error("CDS call 2 dataset is empty. No data was retrieved.")
                return None
            call_2_ds.save(call_2_file)
            file_2_size = os.path.getsize(call_2_file)
            logger.info(f"""Weather data retrieved and saved to '{call_2_file}',
                            File type {type(call_2_file)} and size {file_2_size} bytes""")

            # Read the GRIB files into a DataFrame using _grib_to_df
            data_df_1 = self._grib_to_df(call_1_file)
            if data_df_1 is None: logger.error("Failed to parse the GRIB file, from ek call 1, into a DataFrame.")
            
            data_df_2 = self._grib_to_df(call_2_file)
            if data_df_2 is None: logger.error("Failed to parse the GRIB file from ek call 2, into a DataFrame.")
        
            # seperation of the date (without time) as the new primary index
            ##NOTE: data_df_1 (for variant and invariant parameters) is referenced at time 1200
            data_df_1['date'] = pd.to_datetime(data_df_1['forecast_reference_time']).dt.normalize()
            
            ##NOTE: data_df_2 (for accumulation parameters) is referenced at time 0600
            data_df_2['date'] = pd.to_datetime(data_df_2['forecast_reference_time']).dt.normalize()
            
            # Drop forecast_reference_time before merging
            data_df_1 = data_df_1.drop(columns=['forecast_reference_time'], errors='ignore')
            data_df_2 = data_df_2.drop(columns=['forecast_reference_time'], errors='ignore')  

            # Merge on 'date', 'latitude', and 'longitude'
            ek_df = (
                pd.merge(data_df_1, data_df_2, on=['date', 'latitude', 'longitude'], how='left')
                .sort_values(['date', 'latitude', 'longitude'])
            )
            
            # Move 'date' to the first column
            cols = ['date'] + [col for col in ek_df.columns if col != 'date']
            ek_df = ek_df[cols]

            # Reset the index and save in calling object
            self.ek_dataset = ek_df.reset_index(drop=True)
            
            # Cleanup the temporary files
            os.remove(call_1_file)  # Remove the temporary file
            os.remove(call_2_file)  # Remove the temporary file
            logger.info(f"Temporary GRIB files '{call_1_file}' and '{call_2_file}' have been removed.")
            
            return self.ek_dataset

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error during data retrieval: {e}")
            return None
            
        except Exception as e:
            logger.error(f"Error during data retrieval: {e}")
            # Ensure the temporary file is deleted in case of an error
            if 'call_1_file' in locals() and os.path.exists(call_1_file):
                os.remove(call_1_file)
                logger.info(f"""Temporary GRIB file '{call_1_file}' has been removed due to an error.""")
                return None
            
            if 'call_2_file' in locals() and os.path.exists(call_2_file):
                os.remove(call_2_file)
                logger.info(f"""Temporary GRIB file '{call_2_file}' has been removed due to an error.""")
                return None