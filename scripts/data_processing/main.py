import logging
import pandas as pd
from fb_dataset import FbDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def main():
  data_dir = "scripts/data_processing/raw_data_dir"
  
  # Features added class by class tp start
  eng_feats =            [ # CWFDRS Fire weather indices
                          # 'drought_code', 'duff_moisture_code',
                          # 'fine_fuel_moisture_code', 'initial_spread_index',
                          # 'build_up_index', 'fire_weather_index',
                          
                          # # Weather features
                          # #  - Lightning
                          # 'lightning_products', 'lightning_ratios',
                          # #  - Precipitation
                          # 'rolling_precipitation',
                          # #  - Atmospheric
                          # 'relative_humidity', 'atmospheric_dryness',
                          
                          # # Surface features
                          # #  - Fuel (From vegetation)
                          # 'fuel_low', 'fuel_high',
                          # #  - Soil
                          # 'soil', # Catagorical
                          # #  - Surface water and heat
                          # 'surface_depth_waterheat', # Adds 17 columns, with ratio for each 17cm depth
                          # #  - Topography
                          # 'elevation',
                          # 'x_slope',
                          # 'y_slope'
                          
                          # # Temporal features
                          # # - Seasonal
                          # 'season', 'fire_season',
                          
                          # Spatial features
                          'clusters_12', 'clusters_24', 'clusters_36'
                        ]
  
  ## Initialize  the dataset
  dataset = FbDataset(raw_data_dir=data_dir)
  dataset.config_features(eng_feats=eng_feats) # Can use 'DISABLE' to disable a feature set
  dataset.process() # Load and process data
  processed_data = dataset.get_processed_data() # Get processed data (variable not used because it's stored in the dataset object)
  X_train, X_test, y_train, y_test = dataset.split() # Split data into training and testing sets
  logging.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

  # Save model-ready data to model_data_dir
  model_data_dir = "scripts/modeling/model_data_dir"
  X_train.to_csv(f"{model_data_dir}/X_train.csv", index=False)
  X_test.to_csv(f"{model_data_dir}/X_test.csv", index=False)
  y_train.to_csv(f"{model_data_dir}/y_train.csv", index=False)
  y_test.to_csv(f"{model_data_dir}/y_test.csv", index=False)
  logging.info(f"Model-ready data saved to: {model_data_dir}")


if __name__ == "__main__":
    main()
    
## NOTE:
## Final model should use a sliding window approach, possibly expanding window approach
## Possibly train a seperate model for each timeframe of prediction/forecast (ie. how many days in the future)
## Possibly train a model that predicts the full set of 5 days (ie. a list with days 1-5)

## To implement the sliding window approach, we need to modify the FbDataset class to include a method that generates
## the sliding window data. This method will take in the processed data and return the sliding window data.
    