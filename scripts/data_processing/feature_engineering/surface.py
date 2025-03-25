import numpy as np
import pandas as pd
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    slt = self.raw_data["slt"].copy()
    self.surface_features['soil'] = slt.map(bins)
    
    
    
  ## NOTE: take only sum of temperature volumes in final feature set
  def surface_depth_waterheat(self):
    '''
    Compute surface water-heat ratio from soil water volume and soil temperature.
    Uses depth-specific values and interpolates them to a uniform grid.

    Soil layers:
    - Layer 1: 0 - 7cm
    - Layer 2: 7 - 28cm
    - Layer 3: 28 - 100cm
    - Layer 4: 100 - 289cm
    '''
    
    depths = np.array([7, 28, 100, 289])  # Given soil depths
    resampled_depths = np.linspace(0, 289, 10)  # Target depths
    continuous_x = np.linspace(0, 289, 1000)  # Fine-grained interpolation

    # Convert DataFrame columns to NumPy arrays (shape: [samples, 4])
    swvl = np.stack([self.raw_data['swvl1'], self.raw_data['swvl2'], 
                     self.raw_data['swvl3'], self.raw_data['swvl4']], axis=1)
    
    stl = np.stack([self.raw_data['stl1'], self.raw_data['stl2'], 
                    self.raw_data['stl3'], self.raw_data['stl4']], axis=1)

    # Interpolate for each row (i.e., each sample)
    def interpolate_row(row):
        return np.interp(continuous_x, depths, row)

    continuous_curve_water = np.apply_along_axis(interpolate_row, 1, swvl)
    continuous_curve_temp = np.apply_along_axis(interpolate_row, 1, stl)

    # Resample at evenly spaced intervals for each row
    def resample_row(row):
        return np.interp(resampled_depths, continuous_x, row)

    resampled_swvl = np.apply_along_axis(resample_row, 1, continuous_curve_water)
    resampled_stl = np.apply_along_axis(resample_row, 1, continuous_curve_temp)

    logger.info(f"Resampled SWVL shape: {resampled_swvl.shape}, Resampled STL shape: {resampled_stl.shape}")

    # Compute daily sums (across resampled depths, axis=1)
    daily_swvl_sum = resampled_swvl.sum(axis=1)  # Shape: (num_samples,)
    daily_stl_sum = resampled_stl.sum(axis=1)  # Shape: (num_samples,)

    # Log their shape before adding to the feature set
    logger.info(f"daily_swvl_sum shape: {daily_swvl_sum.shape}, daily_stl_sum shape: {daily_stl_sum.shape}")

    # Store in instance variables
    self.surface_water_sum = daily_swvl_sum
    self.surface_heat_sum = daily_stl_sum

    # Ensure proper Pandas Series assignment
    self.surface_features['daily_water_sum'] = pd.Series(daily_swvl_sum, index=self.raw_data.index)
    self.surface_features['daily_temp_sum'] = pd.Series(daily_stl_sum, index=self.raw_data.index)


    # Compute water-heat ratio
    self.surface_waterheat = np.divide(resampled_swvl, resampled_stl)

    # Store 10 new features in the dataset
    for i in range(10):
        self.surface_features[f'surface_water_heat_{i}'] = self.surface_waterheat[:, i]
    
    
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
    
