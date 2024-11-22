import cdsapi
import pandas as pd
import numpy as np
import xarray as xr  # Import xarray for working with GRIB files
from datetime import timedelta
import requests

class raw_data_manager:
    """A class to manage the raw data collection, combination, and storage process"""

    def __init__(self, cds_client, wildfire_data_path):
        self.cds_client = cds_client
        self.wildfire_data_path = wildfire_data_path

    def load_wildfire_data(self):
        """Load wildfire incidence data from the specified path"""
        wildfire_incidence_data = pd.read_excel(self.wildfire_data_path)
        wildfire_incidence_data['fire_start_date'] = pd.to_datetime(wildfire_incidence_data['fire_start_date'], errors='coerce')
        return wildfire_incidence_data
    
    def filter_fire_dates(self, wildfire_incidence_data):
        """Filter the wildfire incidence data to include only the relevant columns and remove rows with missing values"""
        fire_dates = wildfire_incidence_data[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()
        return fire_dates
    
    def create_all_dates(self, start_date, end_date, freq): ## NOTE: The actual date range to be used in the final combined dataset?
        """Create a DataFrame that contains every nth day between the specified start and end dates"""
        all_dates = pd.date_range(start=start_date, end=end_date, freq=freq).normalize() ## NOTE: ??
        all_dates = all_dates[(all_dates >= pd.Timestamp(start_date)) & (all_dates <= pd.Timestamp(end_date))]
        self.all_dates_df = pd.DataFrame({'date': all_dates})

        return pd.Series(list(set(all_dates).union(fire_dates['fire_start_date']))).sort_values()
    ## NOTE: What is the purpose of line above?

    # def set_variant_variables(self, var_variables):
    #     """Set the time-variant variables for the CDS API request"""
    #     self.variables = variables

    # def set_invariant_variables(self, invar_variables):
    #     """Set the time-invariant variables for the CDS API request"""
    #     self.invariant_variables = invar_variables

    def set_request_parameters(self, var_variables, invar_variables, lat_range, long_range, grid_resolution):
        """Set the parameters for the CDS API request including time-variant variables, time-invariant variables, latitude range, longitude range, and grid resolution"""
        self.var_variables = var_variables # [var1, var2, ...]
        self.invar_variables = invar_variables # [invar1, invar2, ...]
        self.lat_range = lat_range # [lat_min, lat_max]
        self.long_range = long_range # [long_min, long_max]
        self.grid_resolution = grid_resolution # float

    def fetch_weather_data(self, start_date, end_date,  target_file):
        







# # Load wildfire data (ie. wildfire incidence data)
# wildfire_incidence_data = pd.read_excel("scripts/data_collection/fp-historical-wildfire-data-2006-2023.xlsx")

# # Convert 'fire_start_date' to datetime format and extract only the date part
# wildfire_incidence_data['fire_start_date'] = pd.to_datetime(wildfire_data['fire_start_date'], errors='coerce')

# # Filter the fire dates data to only include the relevant columns and remove rows with missing value
# fire_dates = wildfire_incidence_data[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()
# ## NOTE should n/a values be removed? would this be all values where there was no fire?

# # Create a DataFrame that contains every 4th day from 2006 to 2023
# all_dates = pd.date_range(start="2006-01-01", end="2023-12-31", freq='4D').normalize()
# all_dates = pd.Series(list(set(all_dates).union(fire_dates['fire_start_date']))).sort_values()

# # Ensure all dates are within the range 2006-2023
# all_dates = all_dates[(all_dates >= pd.Timestamp("2006-01-01")) & (all_dates <= pd.Timestamp("2023-12-31"))]

# # Create DataFrame for all dates without fire day labels (labeling will be done later)
# all_dates_df = pd.DataFrame({'date': all_dates})
