import pandas as pd
import os

class AbLightningPipeline:
  def __init__(self, data_dir):
    self.data_dir = data_dir
    self.subdirs_list = os.listdir(data_dir)
    
    self.raw_dfs = []
    
    # Collect all csv files in the subdirectories, as dataframes
    for dir in self.subdirs_list:
      for file in os.listfiles(dir):
        if file.endswith('.csv'):
          name = file.split('.')[0]
          df = pd.read_csv(file)
          self.raw_dfs.append({name: df})
          
    self.lightning_df = None      
    
    for df in self.raw_dfs:
      # Merge all dataframes into one
      pass
  
  def _sum_lightning_over_grid(self, grid):
    # Sum lightning strikes and strength over the grid
    pass
  
  def _sum_lightning_over_time(self, period):
    # Sum lightning strikes and strength over time
    pass
  
  def get_lightning_data(self, grid, period):
    # Get lightning data over grid and period
    pass
  
  
        
        
        
    
              
      
