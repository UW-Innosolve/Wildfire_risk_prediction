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
   
    files = [   #2006
                'fb_raw_data_200601.csv', 'fb_raw_data_200602.csv', 'fb_raw_data_200603.csv',
                'fb_raw_data_200604.csv', 'fb_raw_data_200605.csv', 'fb_raw_data_200606.csv',
                'fb_raw_data_200607.csv', 'fb_raw_data_200608.csv', 'fb_raw_data_200609.csv',
                'fb_raw_data_200610.csv', 'fb_raw_data_200611.csv', 'fb_raw_data_200612.csv',
                #2007
                'fb_raw_data_200701.csv', 'fb_raw_data_200702.csv', 'fb_raw_data_200703.csv',
                'fb_raw_data_200704.csv', 'fb_raw_data_200705.csv', 'fb_raw_data_200706.csv',
                'fb_raw_data_200707.csv', 'fb_raw_data_200708.csv', 'fb_raw_data_200709.csv',
                'fb_raw_data_200710.csv', 'fb_raw_data_200711.csv', 'fb_raw_data_200712.csv',
                #2008
                'fb_raw_data_200801.csv', 'fb_raw_data_200802.csv', 'fb_raw_data_200803.csv',
                'fb_raw_data_200804.csv', 'fb_raw_data_200805.csv', 'fb_raw_data_200806.csv',
                'fb_raw_data_200807.csv', 'fb_raw_data_200808.csv', 'fb_raw_data_200809.csv',
                'fb_raw_data_200810.csv', 'fb_raw_data_200811.csv', 'fb_raw_data_200812.csv',
                #2009
                'fb_raw_data_200901.csv', 'fb_raw_data_200902.csv', 'fb_raw_data_200903.csv',
                'fb_raw_data_200904.csv', 'fb_raw_data_200905.csv', 'fb_raw_data_200906.csv',
                'fb_raw_data_200907.csv', 'fb_raw_data_200908.csv', 'fb_raw_data_200909.csv',
                'fb_raw_data_200910.csv', 'fb_raw_data_200911.csv', 'fb_raw_data_200912.csv',
                #2010
                'fb_raw_data_201001.csv', 'fb_raw_data_201002.csv', 'fb_raw_data_201003.csv',
                'fb_raw_data_201004.csv', 'fb_raw_data_201005.csv', 'fb_raw_data_201006.csv',
                'fb_raw_data_201007.csv', 'fb_raw_data_201008.csv', 'fb_raw_data_201009.csv',
                'fb_raw_data_201010.csv', 'fb_raw_data_201011.csv', 'fb_raw_data_201012.csv',
                #2011
                'fb_raw_data_201101.csv', 'fb_raw_data_201102.csv', 'fb_raw_data_201103.csv',
                'fb_raw_data_201104.csv', 'fb_raw_data_201105.csv', 'fb_raw_data_201106.csv',
                'fb_raw_data_201107.csv', 'fb_raw_data_201108.csv', 'fb_raw_data_201109.csv',
                'fb_raw_data_201110.csv', 'fb_raw_data_201111.csv', 'fb_raw_data_201112.csv',
                #2012
                'fb_raw_data_201201.csv', 'fb_raw_data_201202.csv', 'fb_raw_data_201203.csv',
                'fb_raw_data_201204.csv', 'fb_raw_data_201205.csv', 'fb_raw_data_201206.csv',
                'fb_raw_data_201207.csv', 'fb_raw_data_201208.csv', 'fb_raw_data_201209.csv',
                'fb_raw_data_201210.csv', 'fb_raw_data_201211.csv', 'fb_raw_data_201212.csv',
                #2013
                'fb_raw_data_201301.csv', 'fb_raw_data_201302.csv', 'fb_raw_data_201303.csv',
                'fb_raw_data_201304.csv', 'fb_raw_data_201305.csv', 'fb_raw_data_201306.csv',
                'fb_raw_data_201307.csv', 'fb_raw_data_201308.csv', 'fb_raw_data_201309.csv',
                'fb_raw_data_201310.csv', 'fb_raw_data_201311.csv', 'fb_raw_data_201312.csv',
                #2014
                'fb_raw_data_201401.csv', 'fb_raw_data_201402.csv', 'fb_raw_data_201403.csv',
                'fb_raw_data_201404.csv', 'fb_raw_data_201405.csv', 'fb_raw_data_201406.csv',
                'fb_raw_data_201407.csv', 'fb_raw_data_201408.csv', 'fb_raw_data_201409.csv',
                'fb_raw_data_201410.csv', 'fb_raw_data_201411.csv', 'fb_raw_data_201412.csv',
                #2015
                'fb_raw_data_201501.csv', 'fb_raw_data_201502.csv', 'fb_raw_data_201503.csv',
                'fb_raw_data_201504.csv', 'fb_raw_data_201505.csv', 'fb_raw_data_201506.csv',
                'fb_raw_data_201507.csv', 'fb_raw_data_201508.csv', 'fb_raw_data_201509.csv',
                'fb_raw_data_201510.csv', 'fb_raw_data_201511.csv', 'fb_raw_data_201512.csv',
                #2016
                'fb_raw_data_201601.csv', 'fb_raw_data_201602.csv', 'fb_raw_data_201603.csv',
                'fb_raw_data_201604.csv', 'fb_raw_data_201605.csv', 'fb_raw_data_201606.csv',
                'fb_raw_data_201607.csv', 'fb_raw_data_201608.csv', 'fb_raw_data_201609.csv',
                'fb_raw_data_201610.csv', 'fb_raw_data_201611.csv', 'fb_raw_data_201612.csv',
                #2017
                'fb_raw_data_201701.csv', 'fb_raw_data_201702.csv', 'fb_raw_data_201703.csv',
                'fb_raw_data_201704.csv', 'fb_raw_data_201705.csv', 'fb_raw_data_201706.csv',
                'fb_raw_data_201707.csv', 'fb_raw_data_201708.csv', 'fb_raw_data_201709.csv',
                'fb_raw_data_201710.csv', 'fb_raw_data_201711.csv', 'fb_raw_data_201712.csv',
                #2018
                'fb_raw_data_201801.csv', 'fb_raw_data_201802.csv', 'fb_raw_data_201803.csv',
                'fb_raw_data_201804.csv', 'fb_raw_data_201805.csv', 'fb_raw_data_201806.csv',
                'fb_raw_data_201807.csv', 'fb_raw_data_201808.csv', 'fb_raw_data_201809.csv',
                'fb_raw_data_201810.csv', 'fb_raw_data_201811.csv', 'fb_raw_data_201812.csv',
                #2019
                'fb_raw_data_201901.csv', 'fb_raw_data_201902.csv', 'fb_raw_data_201903.csv',
                'fb_raw_data_201904.csv', 'fb_raw_data_201905.csv', 'fb_raw_data_201906.csv',
                'fb_raw_data_201907.csv', 'fb_raw_data_201908.csv', 'fb_raw_data_201909.csv',
                'fb_raw_data_201910.csv', 'fb_raw_data_201911.csv', 'fb_raw_data_201912.csv',
                #2020
                'fb_raw_data_202001.csv', 'fb_raw_data_202002.csv', 'fb_raw_data_202003.csv',
                'fb_raw_data_202004.csv', 'fb_raw_data_202005.csv', 'fb_raw_data_202006.csv',
                'fb_raw_data_202007.csv', 'fb_raw_data_202008.csv', 'fb_raw_data_202009.csv',
                'fb_raw_data_202010.csv', 'fb_raw_data_202011.csv', 'fb_raw_data_202012.csv',
                #2021
                'fb_raw_data_202101.csv', 'fb_raw_data_202102.csv', 'fb_raw_data_202103.csv',
                'fb_raw_data_202104.csv', 'fb_raw_data_202105.csv', 'fb_raw_data_202106.csv',
                'fb_raw_data_202107.csv', 'fb_raw_data_202108.csv', 'fb_raw_data_202109.csv',
                'fb_raw_data_202110.csv', 'fb_raw_data_202111.csv', 'fb_raw_data_202112.csv',
                #2022
                'fb_raw_data_202201.csv', 'fb_raw_data_202202.csv', 'fb_raw_data_202203.csv',
                'fb_raw_data_202204.csv', 'fb_raw_data_202205.csv', 'fb_raw_data_202206.csv',
                'fb_raw_data_202207.csv', 'fb_raw_data_202208.csv', 'fb_raw_data_202209.csv',
                'fb_raw_data_202210.csv', 'fb_raw_data_202211.csv', 'fb_raw_data_202212.csv',
                #2023
                'fb_raw_data_202301.csv', 'fb_raw_data_202302.csv', 'fb_raw_data_202303.csv',
                'fb_raw_data_202304.csv', 'fb_raw_data_202305.csv', 'fb_raw_data_202306.csv',
                'fb_raw_data_202307.csv', 'fb_raw_data_202308.csv', 'fb_raw_data_202309.csv',
                'fb_raw_data_202310.csv', 'fb_raw_data_202311.csv', 'fb_raw_data_202312.csv',
                #2024
                'fb_raw_data_202401.csv', 'fb_raw_data_202402.csv', 'fb_raw_data_202403.csv',
                'fb_raw_data_202404.csv', 'fb_raw_data_202405.csv', 'fb_raw_data_202406.csv',
                'fb_raw_data_202407.csv', 'fb_raw_data_202408.csv', 'fb_raw_data_202409.csv',
                'fb_raw_data_202410.csv', 'fb_raw_data_202411.csv', 'fb_raw_data_202412.csv'
                ]
    
    # for year in range(2006, 2018):
    #     for month in range(1,10):
    #         filename = f"fb_raw_data_{year}0{month}.csv"
    #         files.append(filename)
    #     for month in (10,11,12):
    #         filename = f"fb_raw_data_{year}{month}.csv"
    #         files.append(filename)
    # for year in range(2018, 2019):
    #     for month in range(5,10):
    #         filename = f"fb_raw_data_{year}0{month}.csv"
    #         files.append(filename)
    #     for month in (10,11,12):
    #         filename = f"fb_raw_data_{year}{month}.csv"
    #         files.append(filename)
    # for year in range(2019, 2023):
    #     for month in range(1,10):
    #         filename = f"fb_raw_data_{year}0{month}.csv"
    #         files.append(filename)
    #     for month in (10,11,12):
    #         filename = f"fb_raw_data_{year}{month}.csv"
    #         files.append(filename)
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

complete_raw_data = csv_aggregate("scripts/data_processing/raw_data_dir")
complete_raw_data.to_csv("complete_raw_data.csv", index=False)