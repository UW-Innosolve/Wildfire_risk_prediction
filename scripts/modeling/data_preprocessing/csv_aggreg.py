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
    # files = ['processed_data_no_cffdrs_0.csv', 'processed_data_no_cffdrs_1.csv',
    #          'processed_data_no_cffdrs_2.csv', 'processed_data_no_cffdrs_3.csv',
    #          'processed_data_no_cffdrs_4.csv', 'processed_data_no_cffdrs_5.csv',
    #          'processed_data_no_cffdrs_6_7.csv', 'processed_data_no_cffdrs_8_9_10.csv']
   #  files = ['fb_raw_data_0.csv', 'fb_raw_data_1.csv', 'fb_raw_data_2.csv',
   #           'fb_raw_data_3.csv', 'fb_raw_data_4.csv', 'fb_raw_data_5.csv',
   #           'fb_raw_data_6.csv', 'fb_raw_data_7.csv', 'fb_raw_data_8.csv',
   #           'fb_raw_data_9.csv', 'fb_raw_data_10.csv']#, 'fb_raw_data_11.csv', 'fb_raw_data_12.csv']
    files = []
    for year in range(2006, 2018):
        for month in range(1,10):
            filename = f"fb_raw_data_{year}0{month}.csv"
            files.append(filename)
        for month in (10,11,12):
            filename = f"fb_raw_data_{year}{month}.csv"
            files.append(filename)
    for year in range(2018, 2019):
        for month in range(5,10):
            filename = f"fb_raw_data_{year}0{month}.csv"
            files.append(filename)
        for month in (10,11,12):
            filename = f"fb_raw_data_{year}{month}.csv"
            files.append(filename)
    for year in range(2019, 2023):
        for month in range(1,10):
            filename = f"fb_raw_data_{year}0{month}.csv"
            files.append(filename)
        for month in (10,11,12):
            filename = f"fb_raw_data_{year}{month}.csv"
            files.append(filename)
    # Initialize an empty DataFrame to store the aggregated data
    # data = pd.DataFrame()
    data = []
    # Loop through each file in the directory
    for file in files:
        # Load the data from the file
        df = pd.read_csv(os.path.join(data_dir, file))
        logger.info(f"Loaded {df.shape[0]} rows from {file}")
        # Append the data to the aggregated DataFrame
        data.append(df)
        data_size = sys.getsizeof(data) / 1e6
        logger.info(f"Aggregated data size: {data_size:.2f} MB")
    dataset = pd.concat(data, ignore_index=True)

    final_data_size = sys.getsizeof(data) / 1e6
    logger.info(f"Final aggregated data size: {final_data_size}")
    return dataset
