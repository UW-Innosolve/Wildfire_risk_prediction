import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AbLightningPipeline:
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.subdirs_list = os.scandir(data_dir)
    
    self.raw_dfs = []
    
    # Collect all CSV files from subdirectories
    for dir in self.subdirs_list:
        if dir.is_dir():  # Ensure it's a subdirectory
            for file in os.scandir(dir.path):
                if file.is_file() and file.name.endswith('.csv'):
                    file_path = os.path.join(dir.path, file.name)
                    try:
                        df = pd.read_csv(file_path)
                        logger.info(f"Successfully read: {file_path} ({len(df)} rows)")
                        self.raw_dfs.append(df)
                    except Exception as e:
                        logger.error(f"Failed to read {file_path}: {e}")
                        
    self.lightning_df = self.raw_dfs[0]
              
    for df in self.raw_dfs[1:]:
      if self.lightning_df is None:
        self.lightning_df = pd.DataFrame()
      # Concatenate all dataframes into one
      self.lightning_df = pd.concat([self.lightning_df, df]).sort_values(['date_group', 'latitude', 'longitude'])
      logger.info(f"Merged dataframe of length: {len(df)}, to lightning_df, of length: {len(self.lightning_df)}")

      
  def get_ltng_df(self):
    return self.lightning_df
  
  def _sum_lightning_over_grid(self, grid):
    # Sum lightning strikes and strength over the grid
    pass
  
  def _sum_lightning_over_time(self, period):
    # Sum lightning strikes and strength over time
    pass
  
  def get_lightning_data(self, grid, period):
    # Get lightning data over grid and period
    pass
  
  
        
        
        
    
              
      
