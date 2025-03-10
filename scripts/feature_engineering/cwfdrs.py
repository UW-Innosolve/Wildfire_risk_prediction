from xclim.indices import cwfdrs
import xarray as xr
import numpy as np
import pandas as pd

class FbCwfdrsFeatures():
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df
    self.cwfdrs_features = pd.DataFrame()
    
  def config_features(self):
    self.cwfdrs_features = ['drought_code',
                            'duff_moisture_code',
                            'fine_fuel_moisture_code',
                            'initial_spread_index',
                            'build_up_index',
                            'fire_weather_index',
                            'seasonal_drought_index']
    
  def cwfdrs(self):
    """
    Calculate the Canadian Fire Weather Danger Rating System (CWFDRS) indices.
    """
    # Convert temperature to Celsius
    self.cwfdrs_features['c_temp'] = self.raw_data["2t"] - 273.15
    pass