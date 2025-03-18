## A script to split a large csv into smaller csv files
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_file(csv_dir):
  # Load the large csv file
  logger.info(f"Loading large csv file from {csv_dir}")
  large_csv = pd.read_csv(csv_dir)
  logger.info(f"Loaded large csv file with shape: {large_csv.shape}")

  # Split the large csv file into smaller csv files
  logger.info("Splitting large csv file into smaller csv files")
  for i, chunk in enumerate(pd.read_csv(csv_dir, chunksize=700000)):
    chunk.to_csv(f"parked_data/fb_raw_data_{i}.csv", index=False)
    logger.info(f"Saved chunk {i} with shape: {chunk.shape}")
  logger.info("Finished splitting large csv file")

split_file("scripts/data_processing/raw_data_dir/fb_raw_data_2006-2024.csv")