import numpy as np


class FbSurfaceFeatures():
  def __init__(self, raw_data_df):
    self.raw_data = raw_data_df
    self.surface_features = None
    
    
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
    type_low = self.raw_data['tvl']
    type_high = self.raw_data['tvh']
    
    for veg_type_low, veg_type_high in type_low, type_high:
      if veg_type_low in [1, 10, 7, 16, 17, 19]:
        self.surface_features['fuel_low'] = 1
      else:
        self.surface_features['fuel_low'] = 0
        
      if veg_type_high in [3, 4, 5, 6, 18]:
        self.surface_features['fuel_high'] = 1
      else:
        self.surface_features['fuel_high'] = 0
    
    
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
    
    self.raw_data["slt"] = self.df["soil_catagorical"].map(bins)
    
    
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
    
    ## NOTE: Resampled water and temperature values are not included in the current feature set
    # self.surface_features['resampled_swvl', 'resampled_stl'] = resampled_swvl, resampled_stl
    self.surface_features['soil_temp_to_water'] = resampled_stl / resampled_swvl
    
    
  def topography(self):
    grav_accel = 9.8067
    self.elevation = self.raw_data['z'] / grav_accel
    self.x_slope = np.gradient(self.elevation, self.raw_data['longitude'])
    self.y_slope = np.gradient(self.elevation, self.raw_data['latitude'])
    self.surface_features['elevation', 'x_slope', 'y_slope'] = self.elevation, self.x_slope, self.y_slope
    
