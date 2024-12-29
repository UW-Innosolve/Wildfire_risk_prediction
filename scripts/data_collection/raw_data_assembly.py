
import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RawDataAssembler:
    """A class to manage the raw data assembly and storage process"""

    def __init__(self, wildfire_incidence_data):
        self.wildfire_incidence_data = wildfire_incidence_data
        self.dataset = None

    ## assemble_dataset method
    ##      - assemble the dataset using the specified data pipelines
    ##      - pipelines is a list of dictionaries with the following keys:
    ##          - 'CDS': Copernicus Data Service pipeline
    ##          - Future data pipelines will be added as needed
    ##      - grouping_period_size is the temporal grouping period for the dataset (i.e., temporal size of the output CSVs) 
    ##        must be one of the following:
    ##          - 'D': daily grouping
    ##          - 'W': weekly grouping
    ##          - 'M': monthly grouping
    ##          - 'Y': yearly grouping
    ##      - output: None
    ##      - mutates self.dataset (combining dataframes as they are assembled)
    
    def assemble_dataset(self, pipelines, grouping_period_size, latitude_tolerance=1.0, longitude_tolerance=1.0):
        """Assemble the dataset using the specified data pipelines"""

        logger.info(f"Wildfire Incidence Data Columns in Assembler: {self.wildfire_incidence_data.columns}")

        # Separate fire dates with location data (exclude non-fire dates)
        fire_dates = self.wildfire_incidence_data.dropna(subset=['fire_location_latitude', 'fire_location_longitude'])
        logger.info(f"Number of fire dates with location data: {len(fire_dates)}")

        # Optionally, print first few fire_dates
        logger.debug(f"Sample fire_dates:\n{fire_dates.head()}")

        # Group all dates by the specified period size
        try:
            grouped_wf_data = self.wildfire_incidence_data.groupby(
                self.wildfire_incidence_data['fire_start_date'].dt.to_period(grouping_period_size)
            )
            logger.info(f"Grouped wildfire data by period: {grouping_period_size}")
        except KeyError as e:
            logger.error(f"Grouping failed due to missing key: {e}")
            return

        for pipeline in pipelines:
            if 'CDS' in pipeline:
                cds_pipeline = pipeline['CDS']
                logger.info("CDS pipeline found!")

                for period, batch in grouped_wf_data:
                    time_start = time.time()
                    start_date = batch['fire_start_date'].min()
                    end_date = batch['fire_start_date'].max()
                    logger.info(f"Processing weather data from {start_date} to {end_date}")

                    weather_data = cds_pipeline.fetch_weather_data(start_date, end_date)  # fetch_weather_data returns a DataFrame or None

                    if weather_data is None:
                        logger.error(f"Failed to fetch weather data for period {period}. Skipping this batch.")
                        continue  # Skip to the next batch

                    # Check if 'date' column exists
                    if 'date' not in weather_data.columns:
                        logger.error("Weather data does not contain 'date' column. Skipping this batch.")
                        continue

                    try:
                        weather_data['date'] = weather_data['date'].dt.date
                    except Exception as e:
                        logger.error(f"Error converting 'date' column: {e}")
                        continue

                    # Optionally, print sample weather_data
                    logger.debug(f"Sample weather_data:\n{weather_data[['date', 'latitude', 'longitude']].head()}")

                    logger.info(f"Processing weather data from {start_date} to {end_date}, Data shape: {weather_data.shape}")

                    # Label fire days for the current batch by matching both date and location with a proximity check
                    weather_data['is_fire_day'] = weather_data.apply(
                        lambda row: self._is_fire_labeler(
                            row, 
                            fire_dates,  # Pass filtered fire_dates
                            latitude_tolerance, 
                            longitude_tolerance
                        ), 
                        axis=1
                    )
                    
                    # Check how many fire days were found
                    num_fire_days = weather_data['is_fire_day'].sum()
                    logger.info(f"Number of fire days found in this batch: {num_fire_days}")

                    # Generate the target file name for the weather data
                    target_file = f"weather_data_{period.strftime('%Y%m')}.csv"

                    # Save the DataFrame to a CSV file labeled as above
                    try:
                        weather_data.to_csv(target_file, index=False)
                        logger.info(f"Weather data saved to '{target_file}'.")
                        time_end = time.time()
                        logger.info(f"Processing time for this batch: {time_end - time_start:.2f} seconds")
                    except Exception as e:
                        logger.error(f"Failed to save weather data to '{target_file}': {e}")

    def _is_fire_labeler(self, row, fire_dates, latitude_tolerance, longitude_tolerance):
        """Label the fire incidents in the dataset within a specified location tolerance.
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
            (fire_dates['fire_location_latitude'].between(
                row['latitude'] - latitude_tolerance,
                row['latitude'] + latitude_tolerance
            )) &
            (fire_dates['fire_location_longitude'].between(
                row['longitude'] - longitude_tolerance,
                row['longitude'] + longitude_tolerance
            ))
        ]

        # Debug logging
        if matching_fires.empty:
            logger.debug(f"No fire match found for date {row['date']} and location ({row['latitude']}, {row['longitude']})")
        else:
            logger.debug(f"Fire match found for date {row['date']} and location ({row['latitude']}, {row['longitude']})")

        return int(not matching_fires.empty)