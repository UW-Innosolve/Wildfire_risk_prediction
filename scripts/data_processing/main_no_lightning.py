import logging
import pandas as pd
from fb_dataset import FbDataset
import os
import sys
import traceback
import gc  # for garbage collection

# Create output directory first
output_dir = "scripts/data_processing/processed_data"
os.makedirs(output_dir, exist_ok=True)

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, 'processing.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def process_chunk(chunk, dataset, eng_feats):
    """Process a single chunk of data"""
    try:
        logger.info(f"Processing chunk of shape: {chunk.shape}")
        chunk_dataset = FbDataset(raw_data_dir=None)  # Initialize without loading data
        chunk_dataset.raw_data = chunk
        chunk_dataset.config_features(eng_feats=eng_feats)
        processed_chunk = chunk_dataset.process()
        logger.info(f"Chunk processed successfully. Shape: {processed_chunk.shape}")
        return processed_chunk
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        raise

def main():
    try:
        data_dir = "scripts/data_processing/raw_data_dir"
        logger.info("Starting data processing pipeline")
        
        # Features we can use with available data
        eng_feats = [
            # Weather features
            #  - Precipitation
            'rolling_precipitation',
            #  - Atmospheric
            'relative_humidity', 'atmospheric_dryness',
            
            # Surface features
            #  - Fuel (From vegetation)
            'fuel_low', 'fuel_high',
            #  - Soil
            'soil', # Categorical
            #  - Surface water and heat
            'surface_depth_waterheat', # Adds features for water/heat ratios
            #  - Topography
            'elevation', # From 'z' column
            'slope',
            
            # Temporal features
            # - Seasonal
            'season', 'fire_season',
            
            # Spatial features
            'clusters_12', 'clusters_30'
        ]
        
        logger.info("Loading raw data...")
        # Load data in chunks
        chunk_size = 50000  # Adjust this based on your available memory
        chunks = []
        
        # Read the first chunk to get the total number of rows
        first_chunk = pd.read_csv(os.path.join(data_dir, os.listdir(data_dir)[0]), nrows=1)
        total_rows = sum(1 for file in os.listdir(data_dir) for _ in pd.read_csv(os.path.join(data_dir, file), usecols=[0]))
        logger.info(f"Total rows to process: {total_rows}")
        
        # Process each chunk
        for i in range(0, total_rows, chunk_size):
            logger.info(f"Processing chunk {i//chunk_size + 1} of {(total_rows + chunk_size - 1)//chunk_size}")
            
            # Read chunk
            chunk_data = pd.concat([
                pd.read_csv(os.path.join(data_dir, file), skiprows=range(1, i+1), nrows=chunk_size)
                for file in os.listdir(data_dir)
            ])
            
            if chunk_data.empty:
                break
                
            # Process chunk
            processed_chunk = process_chunk(chunk_data, None, eng_feats)
            chunks.append(processed_chunk)
            
            # Clear memory
            del chunk_data
            gc.collect()
            
            # Save intermediate results
            if len(chunks) > 0:
                intermediate_df = pd.concat(chunks)
                intermediate_df.to_csv(f"{output_dir}/processed_data_intermediate_{i//chunk_size + 1}.csv", index=False)
                logger.info(f"Saved intermediate results for chunk {i//chunk_size + 1}")
                
                # Clear chunks list to free memory
                chunks = []
                gc.collect()
        
        logger.info("Combining all processed chunks...")
        processed_data = pd.concat([pd.read_csv(f"{output_dir}/processed_data_intermediate_{i+1}.csv") 
                                  for i in range((total_rows + chunk_size - 1)//chunk_size)])
        
        logger.info("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = dataset.split(test_size=0.2, random_state=42, apply_smote=True)
        
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        logger.info("Saving final processed data...")
        processed_data.to_csv(f"{output_dir}/processed_data.csv", index=False)
        logger.info(f"Saved processed data to {output_dir}/processed_data.csv")
        
        logger.info("Saving train/test splits...")
        X_train.to_csv(f"{output_dir}/XFt_no_lightning_train.csv", index=False)
        X_test.to_csv(f"{output_dir}/XFt_no_lightning_test.csv", index=False)
        y_train.to_csv(f"{output_dir}/yFt_no_lightning_train.csv", index=False)
        y_test.to_csv(f"{output_dir}/yFt_no_lightning_test.csv", index=False)
        logger.info(f"Saved train/test splits to {output_dir}/")
        
        # Clean up intermediate files
        for i in range((total_rows + chunk_size - 1)//chunk_size):
            os.remove(f"{output_dir}/processed_data_intermediate_{i+1}.csv")
        logger.info("Cleaned up intermediate files")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 