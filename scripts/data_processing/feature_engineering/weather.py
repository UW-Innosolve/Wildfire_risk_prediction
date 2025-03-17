import numpy as np
import pandas as pd

class FbWeatherFeatures():
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df.copy()
    self.weather_features = self.raw_data[['date', 'latitude', 'longitude']]
    
    
  def lightning_features(self):
    self.lightning_products()
    self.lightning_to_ratios()
  
  
  def atmospheric_features(self):
    self.rel_humidity()
    self.rel_atmospheric_dryness()
    
    
  def rel_humidity(self):
    # Relative Humidity calculation
    e_t = np.exp(
        (17.625 * (self.raw_data["2t"] - 273.15))
        / (243.04 + (self.raw_data["2t"] - 273.15))
    )
    e_d = np.exp(
        (17.625 * (self.raw_data["2d"] - 273.15))
        / (243.04 + (self.raw_data["2d"] - 273.15))
    )
    self.weather_features["relative_humidity"] = 100 * (e_d / e_t)
    
    
  def rel_atmospheric_dryness(self):
    # Atmospheric Dryness calculation
    self.weather_features["atmospheric_dryness"] = (
        self.raw_data["2t"] - self.raw_data["2d"]
    ).astype(float)
    
    
  def lightning_products(self):
    # lightning count * multiplicity and lightning count * absolute strength sum
    # self.weather_features["ltng_multiplicity_prod"] = self.raw_data["lightning_count"] * self.raw_data["multiplicity_sum"]
    self.weather_features["ltng_strength_prod"] = self.raw_data["lightning_count"] * self.raw_data["absv_strength_sum"]
    
    
  def lightning_to_ratios(self):
    # lightning count to multiplicity and lightning count to absolute strength sum
    # NOTE: Provides information on the average multiplicity or absolute strength per lightning strike.
    lc = self.raw_data["lightning_count"]
    
    # Set default values for when lightning_count is 0
    # self.weather_features["ltng_multiplicity_ratio"] = self.raw_data["multiplicity_sum"] / lc
    self.weather_features["ltng_strength_ratio"] = self.raw_data["absv_strength_sum"] / lc
    
    # Handle division by zero by replacing NaN values with 0 (lightning_count == 0)
    # self.weather_features["ltng_multiplicity_ratio"].fillna(0, inplace=True)
    self.weather_features["ltng_strength_ratio"].fillna(0, inplace=True)

    
  def rolling_precipitation(self, window=7):
    '''
    Rolling precipitation feature takes the sum of total precipitation over a given window.
    '''
    self.weather_features["rolling_precipitation"] = self.raw_data["tp"].rolling(window=window, min_periods=1).sum()
    
    
  def features(self, lightning=True, atmospheric=True, precipitation=True):
    if lightning:
      self.lightning_features()
    if atmospheric:
      self.atmospheric_features()
    if precipitation:
      self.rolling_precipitation()
    return self.weather_features
  
  
  def get_features(self):
    return self.weather_features
  