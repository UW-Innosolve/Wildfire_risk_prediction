import pandas as pd
import numpy as np
from datetime import timedelta

class RawDataAssembler:
    """A class to manage the raw data assembly and storage process"""

    def __init__(self, wildfire_incidence_data):
        self.wildfire_incidence_data = wildfire_incidence_data
        self.dataset = None

    ## assemble_dataset method
    ##      - assemble the dataset using the specified data pipelines pipelines is a list of data pipelines to be used
    ##      - pipelines is a list of dictionaries with the following keys:
    ##          - 'CDS': Copernicus Data Service pipeline
    ##          - Future data pipelines will be added as needed
    ##      - grouping_period_size is the temporal grouping period for the dataset (ie. temporal size of the output CSVs) must be one of the following:
    ##          - 'D': daily grouping
    ##          - 'W': weekly grouping
    ##          - 'M': monthly grouping
    ##          - 'Y': yearly grouping
    ##      - output: None
    ##      - mutates self.dataset (combining dataframes as they are assembled)
    def assemble_dataset(self, pipelines, grouping_period_size, latitude_tolerance=0.1, longitude_tolerance=0.1):
        """Assemble the dataset using the specified data pipelines"""
        ## NOTE: This method will be able to handle multiple data pipelines, but for now, only the CDS pipeline is implemented
        
        ## Grouping the wildfire incidence data by the specified period size
        grouped_wf_data = self.wildfire_incidence_data.groupby(self.wildfire_incidence_data['fire_start_date'].dt.to_period(grouping_period_size))

        for pipeline in pipelines:
            if 'CDS' in pipelines:
                cds_pipeline = pipeline['CDS']
                print("CDS pipeline found!")
  
                for period, batch in grouped_wf_data:
                    start_date = batch['fire_start_date'].min()
                    end_date = batch['fire_start_date'].max()
                    print(f"Processing weather data from {start_date} to {end_date}")

                    weather_data = cds_pipeline.fetch_weather_data(start_date, end_date) ## fetch_weather_data already returns a dataframe
                    weather_data['date'] = weather_data['date'].dt.date ## Convert 'date' to datetime.date type for matching purposes
                    print(f"Processing weather data from {start_date} to {end_date}, Data shape: {weather_data.shape}")

                    ## Label fire days for the current batch by matching both date and location with a proximity check
                     # Apply the labeling function to the DataFrame
                    weather_data['is_fire_day'] = weather_data.apply(
                        lambda row: self._is_fire_labeler(row, grouped_wf_data, latitude_tolerance, longitude_tolerance), axis=1)
                    
                    # Check how many fire days were found
                    num_fire_days = weather_data['is_fire_day'].sum()
                    print(f"Number of fire days found in this batch: {num_fire_days}")

                    ## Generate the target file name for the weather data
                    target_file = f"weather_data_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"

                    ## Save the DataFrame to a CSV file labled as above
                    weather_data.to_csv(target_file, index=False)


    ## _is_fire_labeler method
    ##      - label the fire incidents in the dataset within a specified location tolerance
    ##      - intended to be used with apply
    ##      - input: row, fire_dates, latitude_tolerance, longitude_tolerance
    ##      - output: 1 if a matching fire is found, 0 otherwise
    ##      - private method
    def _is_fire_labeler(self, row, fire_dates, latitude_tolerance, longitude_tolerance):
        """Label the fire incidents in the dataset within a specified location tolerance.
        This method is intended to be used with `apply`.
        Args:
            row: A single row of the DataFrame (supplied automatically by `apply`).
            fire_dates: DataFrame containing fire incident data with columns:
                - fire_start_date
                - fire_location_latitude
                - fire_location_longitude
            latitude_tolerance: Latitude tolerance for matching locations.
            longitude_tolerance: Longitude tolerance for matching locations.

        Returns:
            int: 1 if a matching fire is found, 0 otherwise.
        """
        matching_fires = fire_dates[
            (fire_dates['fire_start_date'] == row['date']) &
            (fire_dates['fire_location_latitude'].between(row['latitude'] - latitude_tolerance,
                                                        row['latitude'] + latitude_tolerance)) &
            (fire_dates['fire_location_longitude'].between(row['longitude'] - longitude_tolerance,
                                                        row['longitude'] + longitude_tolerance))]
        
        # debug printing
        if matching_fires.empty:
            print(f"No fire match found for date {row['date']} and location ({row['latitude']}, {row['longitude']})")
        else:
            print(f"Fire match found for date {row['date']} and location ({row['latitude']}, {row['longitude']})")

        return int(not matching_fires.empty)
