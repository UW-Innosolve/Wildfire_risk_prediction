import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.info)
logger = logging.getLogger(__name__)

class AbLightningPipeline:
  def __init__(self, data_dir):
    self.final_lightning_df = None
    
    self.lat_range = None
    self.lon_range = None
    self.grid_resolution = None
    
    self.data_dir = data_dir
    self.subdirs_list = os.scandir(data_dir)
    
    self.raw_dfs_from_csv = []
    
    # Check if the merged CSV file exists
    if 'ab_lightning_data.csv' in os.listdir(data_dir):
      logger.info("Merged Lightning CSV file exists. Reading from file.")
      self.raw_lightning_df = pd.read_csv(os.path.join(data_dir, 'ab_lightning_data.csv'))
      
    else:
      # Collect all CSV files from subdirectories
      for dir in self.subdirs_list:
          if dir.is_dir():  # Ensure it's a subdirectory
              for file in os.scandir(dir.path):
                  if file.is_file() and file.name.endswith('.csv'):
                      file_path = os.path.join(dir.path, file.name)
                      try:
                          df = pd.read_csv(file_path)
                          logger.info(f"Successfully read: {file_path} ({len(df)} rows)")
                          self.raw_dfs_from_csv.append(df)
                      except Exception as e:
                          logger.error(f"Failed to read {file_path}: {e}")
                          
      self.raw_lightning_df = self.raw_dfs_from_csv[0]
                
      for df in self.raw_dfs_from_csv[1:]:
        if self.raw_lightning_df is None:
          self.raw_lightning_df = pd.DataFrame()
        # Concatenate all dataframes into one
        self.raw_lightning_df = pd.concat([self.raw_lightning_df, df]).sort_values(['date_group', 'latitude', 'longitude'])
        logger.info(f"Merged dataframe of length: {len(df)}, to lightning_df, of length: {len(self.raw_lightning_df)}")
      
      self.raw_lightning_df['date'] = pd.to_datetime(self.raw_lightning_df['date_group'], format='mixed')
        
      # Save the merged dataframe to a CSV file
      self.raw_lightning_df.to_csv(os.path.join(data_dir, 'ab_lightning_data.csv'), index=False)


  def get_raw_ltng_df(self):
    return self.raw_lightning_df
  
  
  def set_ab_ltng_params(self, lat_range, lon_range, grid_resolution):
    self.lat_range = lat_range
    self.lon_range = lon_range
    self.grid_resolution = grid_resolution
    self.grid = self._generate_grid(lat_range, lon_range, grid_resolution)
    
    
  def _generate_grid(self, lat_range, lon_range, grid_resolution):
    grid = pd.DataFrame(columns=['latitude', 'longitude'])
    
    lat_values = np.arange(lat_range[0], lat_range[1], grid_resolution)
    lon_values = np.arange(lon_range[0], lon_range[1], grid_resolution)

    # Generate grid
    grid = pd.DataFrame(
      [(lat, lon) for lat in lat_values for lon in lon_values],
      columns=['latitude', 'longitude']
    )
    
    return grid
  
  
  ## Mutates final_lightning_df
  ## NOTE: _generlate_grid must be called before this function
  def _sum_ltng(self, final_ltng_df):
    raw_ltng_df = self.get_raw_ltng_df()
    
    # Date slice the raw lightning data for the batch
    max_date = final_ltng_df['date'].max()
    min_date = final_ltng_df['date'].min()
    batch_dated_raw_ltng_df = raw_ltng_df[
      (pd.to_datetime(raw_ltng_df['date']) >= pd.to_datetime(min_date)) &
      (pd.to_datetime(raw_ltng_df['date']) <= pd.to_datetime(max_date))]
    
    # Iterate over each lightning event in the batch
    for _, row in batch_dated_raw_ltng_df.iterrows():
      lat, lon = row['latitude'], row['longitude']
      logger.info(f"Processing lightning event at lat: {lat}, lon: {lon}, from date: {row['date_group']}")
      # Find the grid cell that this lightning event belongs to
      grid_cell = self.grid[(self.grid['latitude'] <= lat) & (self.grid['latitude'] + self.grid_resolution > lat) &
                (self.grid['longitude'] <= lon) & (self.grid['longitude'] + self.grid_resolution > lon)]
      
      if not grid_cell.empty: # If the lightning event is within the grid
        grid_lat, grid_lon = grid_cell.iloc[0]['latitude'], grid_cell.iloc[0]['longitude']
        # Find the corresponding row in final_lightning_df
        final_row = final_ltng_df[(final_ltng_df['latitude'] == grid_lat) &
                    (final_ltng_df['longitude'] == grid_lon) &
                    (final_ltng_df['date'] == row['date_group'])]
        
        if not final_row.empty:
          idx = final_row.index[0]
          final_ltng_df.at[idx, 'lightning_count'] += 1
          final_ltng_df.at[idx, 'absv_strength_sum'] += np.abs(row['strength'])
          final_ltng_df.at[idx, 'multiplicity_sum'] += row['multiplicity']

  
  def get_ltng_data(self, batch_dates):
    self.final_lightning_df = pd.DataFrame(columns=['date', 'latitude', 'longitude', 'lightning_count', 'absv_strength_sum', 'multiplicity_sum'])
    for date in batch_dates:
      logger.info(f"Generating empty grid values for date: {date}")
      self.final_lightning_df = pd.concat([self.final_lightning_df, pd.DataFrame({'date': date,
                                                                                  'latitude': self.grid['latitude'],
                                                                                  'longitude': self.grid['longitude'],
                                                                                  'lightning_count': 0,
                                                                                  'absv_strength_sum': 0,
                                                                                  'multiplicity_sum': 0})])
      
    self._sum_ltng(self.final_lightning_df)
    return self.final_lightning_df
  
        
        
        
    
              
      
