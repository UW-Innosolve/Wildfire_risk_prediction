## A script that checks a csv file for null values and prints out the number of null values in each column
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csv_dir = "scripts/data_processing/processed_data_dir/processed_data.csv"

def null_check(csv_dir):
    """
    Check for null values in a csv file and print out the number of null values in each column.
    """
    logger.info(f"Checking for null values in {csv_dir}")
    data = pd.read_csv(csv_dir)
    null_values = data.isnull().sum()
    logger.info(f"Null values in {csv_dir}:")
    logger.info(null_values)
    
null_check(csv_dir)