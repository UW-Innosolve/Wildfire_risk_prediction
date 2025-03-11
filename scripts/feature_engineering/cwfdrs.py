from xclim.indices import cwfdrs
import xarray as xr
import numpy as np
import pandas as pd

class FbCwfdrsFeatures():
  def __init__(self, raw_data_df):
    """
    Initialize the class with raw input data (Pandas DataFrame).
    Convert it to an xarray.Dataset for processing.
    """
    self.raw_data = raw_data_df
    self.cwfdrs_inputs = self._convert_to_xarray(raw_data_df)
    self.cwfdrs_features = pd.DataFrame()
    
    self.config_features()
    self.compute_cwfdrs()

  def _convert_to_xarray(self, df):
    """
    Convert the Pandas DataFrame into an xarray.Dataset.
    Assumes 'date', 'latitude', and 'longitude' columns exist.
    """
    return df.set_index(["date", "latitude", "longitude"]).to_xarray()

  def config_features(self):
    """Define the feature names for the dataset."""
    self.cwfdrs_features = ['drought_code',
                            'duff_moisture_code',
                            'fine_fuel_moisture_code',
                            'initial_spread_index',
                            'build_up_index',
                            'fire_weather_index']

  def compute_cwfdrs(self):
    """
    Calculate the Canadian Fire Weather Danger Rating System (CWFDRS) indices.
    """
    # Convert temperature to Celsius
    self.cwfdrs_inputs["tas"] = self.cwfdrs_inputs["2t"] - 273.15  # Kelvin to Celsius

    # Compute relative humidity from dew point temperature
    self.cwfdrs_inputs["hurs"] = cwfdrs.relative_humidity(
        tas=self.cwfdrs_inputs["tas"],
        tdps=self.cwfdrs_inputs["2d"] - 273.15
    )

    # Compute wind speed from U/V vector components
    self.cwfdrs_inputs["sfcWind"] = cwfdrs.wind_speed(
        u=self.cwfdrs_inputs["10u"],
        v=self.cwfdrs_inputs["10v"]
    )

    # Assign precipitation
    self.cwfdrs_inputs["pr"] = self.cwfdrs_inputs["sf"]  # Precipitation in mm/day

    # Extract latitude (assumed to be constant across dataset)
    self.cwfdrs_inputs["lat"] = self.cwfdrs_inputs["latitude"]

    # Compute Fire Weather Indices
    fire_indices = cwfdrs.cffwis_indices(
        tas=self.cwfdrs_inputs["tas"],
        pr=self.cwfdrs_inputs["pr"],
        hurs=self.cwfdrs_inputs["hurs"],
        sfcWind=self.cwfdrs_inputs["sfcWind"],
        lat=self.cwfdrs_inputs["lat"],
        overwintering=True,
        dry_start="CFS"
    )

    # Convert back to Pandas DataFrame
    self.cwfdrs_features = fire_indices.to_dataframe().reset_index()
      
  def get_features(self):
    """Return the computed CWFDRS features."""
    return self.cwfdrs_features

# from xclim.indices import cwfdrs
# import xarray as xr
# import numpy as np
# import pandas as pd

# class FbCwfdrsFeatures():
#   def __init__(self, raw_data_df):
#     self.raw_data = raw_data_df
#     self.cwfdrs_inputs = xr.Dataset()
#     self.cwfdrs_features = pd.DataFrame()
    
#   def config_features(self):
#     self.cwfdrs_features = ['drought_code',
#                             'duff_moisture_code',
#                             'fine_fuel_moisture_code',
#                             'initial_spread_index',
#                             'build_up_index',
#                             'fire_weather_index',
#                             'seasonal_drought_index']
    
#   def cwfdrs(self):
#     """
#     Calculate the Canadian Fire Weather Danger Rating System (CWFDRS) indices.
#     """
#     # Convert temperature to Celsius
#     self.cwfdrs_inputs['c_temp'] = self.raw_data["2t"] - 273.15
#     self.cwfdrs_inputs['relative_humidity'] = cwfdrs.relative_humidity(
#       tas=self.cwfdrs_inputs['c_temp'],
#       tdps=self.raw_data["2d"] - 273.15
#     )
#     self.cwfdrs_inputs['wind_speed'] = cwfdrs.wind_speed(
#       u=self.raw_data["10u"],
#       v=self.raw_data["10v"]
#     )
#     self.cwfdrs_inputs['precipitation'] = self.raw_data["sf"]
#     self.cwfdrs_inputs['latitude'] = self.raw_data["latitude"]
#     self.cwfdrs_inputs['longitude'] = self.raw_data["longitude"]
    
    


# from xclim.indices import cwfdrs
# import xarray as xr
# import numpy as np
# import pandas as pd

# class FbCwfdrsFeatures():
#   def __init__(self, raw_data_df):
#     self.raw_data = raw_data_df
#     self.cwfdrs_features = pd.DataFrame()
    
#   def config_features(self):
#     self.cwfdrs_features = ['drought_code',
#                             'duff_moisture_code',
#                             'fine_fuel_moisture_code',
#                             'initial_spread_index',
#                             'build_up_index',
#                             'fire_weather_index',
#                             'seasonal_drought_index']
    
#   def cwfdrs(self):
#     """
#     Calculate the Canadian Fire Weather Danger Rating System (CWFDRS) indices.
#     """
#     # Convert temperature to Celsius
#     self.cwfdrs_features['c_temp'] = self.raw_data["2t"] - 273.15
#     pass