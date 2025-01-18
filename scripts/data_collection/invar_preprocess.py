## This file contains classes related to the loading and pre-processing of invariant data.
import logging
from cds_pipeline import CdsPipeline
import pandas as pd


class InvarDataLoader:
    def __init__(self):
        pass

    def load_invar_data(self, invar_data_path):
        """
        Load invariant data from csv file at path.
        """
        invar_data = pd.read_csv(invar_data_path)
        return invar_data
    
    def create_space_df(self, lat_range, long_range, grid_resolution):
        """
        Create a space-time dataframe for time-invariant data.
        """
        latitudes = list(range(lat_range[0], lat_range[1], grid_resolution))
        longitudes = list(range(long_range[0], long_range[1], grid_resolution))
        invar_data = pd.DataFrame()
        for lat in latitudes:
            for lon in longitudes:
                invar_data = invar_data.append({'latitude': lat, 'longitude': lon}, ignore_index=True)
        return invar_data
    
    def replicate_across_time(self, invar_data, start_date, end_date):
        """
        Replicate the invariant data across time.
        """
        pass