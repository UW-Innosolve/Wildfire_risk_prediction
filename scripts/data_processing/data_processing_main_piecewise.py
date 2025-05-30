import logging
import pandas as pd
import os
import logging
from fb_dataset import FbDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def main():
    # Paths for input and output folders
  input_folder = "scripts/data_processing/raw_subset"
  output_folder = "scripts/data_processing/processed_data_dir"

  # Ensure output folder exists
  os.makedirs(output_folder, exist_ok=True)

  # Get sorted list of CSV files from both folders
  input_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])
  
  eng_feats =  [
                # Weather features
                #  - Lightning
                'lightning_products', 'lightning_ratios',
                #  - Precipitation
                'rolling_precipitation',
                #  - Atmospheric
                'relative_humidity', 'atmospheric_dryness',

                # Surface features
                #  - Fuel (From vegetation)
                'fuel_low', 'fuel_high',
                #  - Soil
                'soil', # Catagorical
                # #  - Surface water and heat
                'surface_depth_waterheat', # Adds 10 features (columns)
                #  - Topography
                'elevation',
                'slope',

                # Temporal features
                # - Seasonal
                'season', 'fire_season',

                # Spatial features
                'clusters_12', 'clusters_30'
              ]

  # Loop through files that are present in both folders
  for month_file in input_files:
    month_file_path = os.path.join(input_folder, month_file)
    ## Initialize  the dataset
    dataset = FbDataset(raw_data_dir=month_file_path, one_file_at_a_time=True)
    dataset.config_features(eng_feats=eng_feats) # Can use 'DISABLE' to disable a feature set
    dataset.process() # Load and process data
    processed_data = dataset.get_processed_data() # Get processed data (variable not used because it's stored in the dataset object)
    
    ## Check entire processed_data for NaN values
    if processed_data.isnull().values.any():
        logger.warning(f"NaN values found in processed data for month: {month_file}")
        logger.warning(f"Number of NaN values: {processed_data.isnull().sum().sum()}")
        logger.warning(f"Columns with NaN values: {processed_data.columns[processed_data.isnull().any()]}")
        logger.warning(f"Rows with NaN values: {processed_data[processed_data.isnull().any(axis=1)]}")
    else:
        logger.info(f"No NaN values found in processed data for month: {month_file}")
    
    logger.info(f"Processed data shape: {processed_data.shape}")
    month_file_path_output = os.path.join(output_folder, ("processed_" + month_file))
    processed_data.to_csv(month_file_path_output, index=False)
    logger.info(f"Processed data saved to: {month_file_path_output}")
    dataset.destroy()



if __name__ == "__main__":
    main()
    
## NOTE:
## Final model should use a sliding window approach, possibly expanding window approach
## Possibly train a seperate model for each timeframe of prediction/forecast (ie. how many days in the future)
## Possibly train a model that predicts the full set of 5 days (ie. a list with days 1-5)

## To implement the sliding window approach, we need to modify the FbDataset class to include a method that generates
## the sliding window data. This method will take in the processed data and return the sliding window data.
    