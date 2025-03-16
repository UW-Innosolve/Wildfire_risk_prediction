import numpy as np
import pandas as pd


class FbSurfaceFeatures():
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df
    self.surface_features = self.raw_data[['date', 'latitude', 'longitude']]
    
    
  def vegetation(self):
    '''
    Fuel type catagories (preliminary):
    
    fuel_high includes any of the following:
    - 3 Evergreen Needleleaf Forest
    - 4 Deciduous Needleleaf Forest
    - 5 Evergreen Broadleaf Forest
    - 6 Deciduous Broadleaf Forest
    - 18 Mixed Forest/woodland
    
    fuel_low includes any of the following:
    - 1 crops/mixed farming
    - 10 irrigated croplands
    - 7 tall grass
    - 16 evergreen shrubland
    - 17 deciduous shrubland
    - 19 interupted forest
    
    no_fuel is and vegetation type not listed above
    '''
    # Define vegetation categories
    low_fuel_types = {1, 10, 7, 16, 17, 19}
    high_fuel_types = {3, 4, 5, 6, 18}

    # Vectorized operations to assign 1 or 0 based on conditions
    self.surface_features["fuel_low"] = self.raw_data["tvl"].isin(low_fuel_types).astype(int)
    self.surface_features["fuel_high"] = self.raw_data["tvh"].isin(high_fuel_types).astype(int)
        
      
    
    
  def soil(self):
    """
    Categorizes soil types 'slt' into three bins: 'Coarse', 'Medium', and 'Organic'.
    """
    bins = {
        1: "Coarse",
        2: "Coarse",
        3: "Medium",
        4: "Medium",
        5: "Medium",
        6: "Organic",
        7: "Organic"
    }
    
    self.surface_features['soil'] = self.raw_data["slt"].map(bins)
    
    
    ## NOTE: take only sum of temperature volumes in final feature set
  def surface_depth_waterheat(self):
    '''
    layer 1: 0 - 7cm
    layer 2: 7 - 28cm
    layer 3: 28 - 100cm
    layer 4: 100 - 289cm
    '''
    # Define the depths of the soil layers
    depths = [0, 7, 28, 100, 289]
    resampled_depths = np.arange(0, 289, 17)

    ## Resample the soil water volume at 17cm intervals
    swvl = [self.raw_data['swvl1'], self.raw_data['swvl2'], self.raw_data['swvl3'], self.raw_data['swvl4']]
    continuous_curve_water = np.interp(np.arange(0, 289, 1), depths[:-1], swvl)
    
    # Resample at an evenly spaced interval of 17cm
    resampled_swvl = np.interp(resampled_depths, np.arange(0, 289, 1), continuous_curve_water)
    
    ## Resample the soil temperature at 17cm intervals
    stl1 = [self.raw_data['stl1'], self.raw_data['stl2'], self.raw_data['stl3'], self.raw_data['stl4']]
    continuous_curve_temp = np.interp(np.arange(0, 289, 1), depths[:-1], stl1)
    
    # Resample at an evenly spaced interval of 17cm
    resampled_stl = np.interp(resampled_depths, np.arange(0, 289, 1), continuous_curve_temp)
    
    daily_swvl_sum = resampled_swvl.sum()
    daily_stl_sum = resampled_stl.sum()
    ## NOTE: Resampled water and temperature values are not included in the current feature set
    # self.surface_features['resampled_swvl', 'resampled_stl'] = resampled_swvl, resampled_stl
    # self.surface_features['surface_water_heat'] = resampled_stl / resampled_swvl
    
    self.surface_water_sum = daily_swvl_sum
    self.surface_heat_sum = daily_stl_sum
    self.surface_features['daily_water_sum', 'daily_temp_sum'] = daily_swvl_sum, daily_stl_sum
    self.surface_features['surface_water_heat'] = resampled_stl / resampled_swvl
    
  def topography(self):
    grav_accel = 9.8067
    lat_meters = 111320  # Meters per degree latitude
    lon_meters = 111320 * np.cos(np.radians(self.raw_data["latitude"]))  # Meters per longitude degree

    # Normalize elevation
    self.elevation = self.raw_data['z'] / grav_accel
    
    # Prevent division by zero issues
    if self.raw_data["longitude"].nunique() == 1:  # Constant longitudes
        self.x_slope = np.zeros_like(self.elevation)
    else:
        self.x_slope = np.gradient(self.elevation, self.raw_data['longitude'] * lon_meters)

    if self.raw_data["latitude"].nunique() == 1:  # Constant latitudes
        self.y_slope = np.zeros_like(self.elevation)
    else:
        self.y_slope = np.gradient(self.elevation, self.raw_data['latitude'] * lat_meters)
        
    # Fix extreme values
    self.x_slope = np.nan_to_num(self.x_slope, nan=0.0, posinf=0.0, neginf=0.0)
    self.y_slope = np.nan_to_num(self.y_slope, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute slope magnitude safely
    self.slope = np.sqrt(self.x_slope**2 + self.y_slope**2)

    # Store results
    self.surface_features['elevation'] = self.elevation
    self.surface_features['slope'] = self.slope
    # grav_accel = 9.8067
    # self.elevation = self.raw_data['z'] / grav_accel
    # self.x_slope = np.gradient(self.elevation, self.raw_data['longitude'])
    # self.y_slope = np.gradient(self.elevation, self.raw_data['latitude'])
    # self.slope = np.sqrt(self.x_slope**2 + self.y_slope**2)
    
    # self.surface_features['elevation'] = self.elevation
    # self.surface_features['slope'] = self.slope
    
  def get_features(self):
    return self.surface_features
    
