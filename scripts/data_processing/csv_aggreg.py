# Simple script to aggregate multple csvs and outputs it to a dataframe
import pandas as pd
import numpy as np
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def csv_aggregate(data_dir):
    """
    Aggregate multiple csv files in a directory into a single DataFrame.
    """
    # Get the list of files in the directory
    files = os.listdir(data_dir)
    # Initialize an empty DataFrame to store the aggregated data
    data = pd.DataFrame()
    # Loop through each file in the directory
    for file in files:
        # Load the data from the file
        df = pd.read_csv(os.path.join(data_dir, file))
        logger.info(f"Loaded {df.shape[0]} rows from {file}")
        # Append the data to the aggregated DataFrame
        data = data.append(df)
        data_size = sys.getsizeof(data) / 1e6
        logger.info(f"Aggregated data size: {data_size:.2f} MB")
        
    final_data_size = sys.getsizeof(data) / 1e6
    logger.info(f"Final aggregated data size: {final_data_size}")
    return data

