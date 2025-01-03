import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RawDataAssembler:
    """A class to manage the raw data assembly and storage process.
        Must be initialized with:
            - wildfire_incidence_data: DataFrame containing wildfire incidence data with columns:
            - start_date: Start date for the dataset
            - end_date: End date for the dataset
            - resample_interval: Interval for resampling the dataset (e.g., '4D' is every 4th day)
            - grouping_period_size: Temporal grouping period for the dataset (i.e., temporal size of the output CSVs)
                grouping_period_size must be one of the following:
                    - 'D': daily grouping
                    - 'W': weekly grouping
                    - 'M': monthly grouping
                    - 'Y': yearly grouping
            - latitude_tolerance: Latitude tolerance for matching locations
            - longitude_tolerance: Longitude tolerance for matching locations
    """

    def __init__(self, wildfire_incidence_data,
                 start_date, end_date, resample_interval,
                 grouping_period_size, latitude_tolerance, longitude_tolerance):

        # Load wildfire incidence data and separate fire dates with location data
        self.wildfire_incidence_data = wildfire_incidence_data
        self.fire_dates = self.wildfire_incidence_data[
            ['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']
        ].dropna()

        # Ensure fire_start_date is of type datetime.date for matching purposes
        self.fire_dates['fire_start_date'] = self.fire_dates['fire_start_date'].dt.date
        logger.info(f"Number of fire dates with location data: {len(self.fire_dates)}")
        logger.debug(f"Sample fire_dates:\n{self.fire_dates.head()}")

        # Set the start and end dates for the dataset
        self.start_date = start_date
        self.end_date = end_date

        # Set the resample interval for the dataset
        self.resample_interval = resample_interval

        # Set the temporal grouping period for the dataset
        self.grouping_period_size = grouping_period_size

        # Set the latitude and longitude tolerance for matching fire locations
        self.latitude_tolerance = latitude_tolerance
        self.longitude_tolerance = longitude_tolerance

        # Initialize the variable to store the dataset
        self.dataset = None

    ## assemble_dataset method
    ##      - assemble the dataset using the specified data pipelines
    ##      - pipelines is a list of dictionaries with keys like 'CDS', 'HUMAN_ACTIVITY', etc.
    ##      - grouping_period_size is the temporal grouping period (e.g., 'M' for monthly)
    ##      - output: None
    ##      - mutates self.dataset (combining dataframes as they are assembled)
    def assemble_dataset(self, pipelines):
        """Assemble the dataset using the specified data pipelines"""
        logger.info(f"Wildfire Incidence Data Columns in Assembler: {self.fire_dates.columns}")
        logger.info(f"Pipeline list: {pipelines}")

        # Generate a DataFrame with all dates (fire and non-fire) for the specified period
        self.all_dates_df = self._all_dates_generator(
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.resample_interval,
            fire_dataset=self.fire_dates
        )

        # Group all dates by the specified period size
        try:
            self.grouped_all_dates = self.all_dates_df.groupby(
                self.all_dates_df['date'].dt.to_period(self.grouping_period_size)
            )
            logger.info(f"Grouped all_dates data by period: {self.grouping_period_size}")
            logger.debug(f"Sample grouped_all_dates:\n{self.grouped_all_dates.head()}")
        except KeyError as e:
            logger.error(f"Grouping failed due to missing key: {e}")
            return

        # Iterate over each period (e.g., month)
        for period, batch in self.grouped_all_dates:
            # Start timing for this batch
            time_start = time.time()

            logger.info(f"--- Processing batch for period: {period} ---")
            logger.info(f"Batch shape: {batch.shape}")

            start_date = batch['date'].min()
            end_date = batch['date'].max()
            period_key = period.strftime('%Y%m')

            # Initialize a DataFrame to hold the integrated data for this period
            monthly_data = None

            # Process each pipeline in sequence
            for pipeline in pipelines:
                # 1) CDS pipeline
                if 'CDS' in pipeline:
                    cds_pipeline = pipeline['CDS']
                    logger.info("CDS pipeline found!")

                    # Fetch weather data
                    logger.info(f"Starting request for weather data from {start_date} to {end_date}")
                    weather_data = cds_pipeline.fetch_weather_data(start_date, end_date)

                    if weather_data is None or weather_data.empty:
                        logger.error(f"Failed to fetch weather data for period {period_key}. Skipping.")
                        continue

                    # Check if 'date' column exists
                    if 'date' not in weather_data.columns:
                        logger.error("Weather data does not contain 'date' column. Skipping this batch.")
                        continue

                    logger.info(f"Processing weather data from {start_date} to {end_date}, Data shape: {weather_data.shape}")

                    # Label fire days in weather data
                    logger.info("Labeling fire days in weather data...")
                    weather_data['is_fire_day'] = weather_data.apply(self._is_fire_labeler, axis=1)
                    num_fire_days = weather_data['is_fire_day'].sum()
                    logger.info(f"Number of fire days found in this batch: {num_fire_days}")

                    # Store the resulting DataFrame in monthly_data
                    monthly_data = weather_data.copy()

                # 2) HUMAN_ACTIVITY pipeline
                elif 'HUMAN_ACTIVITY' in pipeline:
                    if monthly_data is None or monthly_data.empty:
                        logger.warning(f"No monthly_data from CDS to integrate with HUMAN_ACTIVITY for {period_key}.")
                        continue

                    ha_pipeline = pipeline['HUMAN_ACTIVITY']
                    logger.info("HumanActivity pipeline found!")

                    # Fetch and integrate Human Activity data
                    monthly_data = ha_pipeline.fetch_human_activity_monthly(monthly_data, period_key)
                    logger.info(f"Integrated HumanActivity data into {period_key} => final shape={monthly_data.shape}")

                else:
                    # If there are other pipelines (e.g., LIGHTNING), they'd go here:
                    logger.warning(f"Unknown pipeline key in {pipeline}. Skipping this pipeline.")

            # After all pipelines are processed for this period, write the final CSV
            if monthly_data is not None and not monthly_data.empty:
                target_file = f"final_weather_data_{period_key}.csv"
                try:
                    monthly_data.to_csv(target_file, index=False)
                    logger.info(f"Wrote monthly CSV for {period_key} -> '{target_file}'")
                except Exception as e:
                    logger.error(f"Failed to save updated weather data for {period_key}: {e}")
            else:
                logger.warning(f"No final data to save for {period_key}.")

            # End timing for this batch
            time_end = time.time()
            logger.info(f"Processing time for this batch ({period_key}): {time_end - time_start:.2f} seconds")

    ## _is_fire_labeler method
    ##      - label the fire incidents in the dataset within a specified location tolerance
    ##      - input: row, fire_dates, latitude_tolerance, longitude_tolerance
    ##      - output: 1 if a matching fire is found, 0 otherwise
    def _is_fire_labeler(self, row):
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
        matching_fires = self.fire_dates[
            (self.fire_dates['fire_start_date'] == row['date']) &
            (self.fire_dates['fire_location_latitude'].between(
                row['latitude'] - self.latitude_tolerance,
                row['latitude'] + self.latitude_tolerance
            )) &
            (self.fire_dates['fire_location_longitude'].between(
                row['longitude'] - self.longitude_tolerance,
                row['longitude'] + self.longitude_tolerance
            ))
        ]
        return int(not matching_fires.empty)

    ## all_dates_generator
    ##      - generates a dataframe of dates with all fire dates in fire_dataset and non-fire dates resampled to the specified interval
    ##      - input: start_date, end_date, interval (e.g., '4D' is every 4th day), fire_dataset
    ##      - output: all_dates_df (pandas DataFrame)
    def _all_dates_generator(self, start_date, end_date, interval, fire_dataset):
        fire_dates = fire_dataset[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()

        all_dates = pd.date_range(start=start_date, end=end_date, freq=interval).normalize()

        fire_dates['fire_start_date'] = pd.to_datetime(fire_dates['fire_start_date'])

        all_dates = pd.Series(list(set(all_dates).union(fire_dates['fire_start_date']))).sort_values()

        all_dates = all_dates[(all_dates >= pd.Timestamp(start_date)) & (all_dates <= pd.Timestamp(end_date))]

        all_dates_df = pd.DataFrame({'date': all_dates})
        logger.info(f"all_dates count (constructed from fire_dates + every nth (interval) day): {len(all_dates_df)}")
        logger.debug(f"Sample all_dates:\n{all_dates_df.head()}")

        return all_dates_df
