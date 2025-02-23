import pandas as pd
import numpy as np
import datetime as dt
import os
import time
import logging
# from oapi_pipeline.human_activity_pipeline import HumanActivityPipeline as hap

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
        self.fire_dates = self.wildfire_incidence_data[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()
        
        # Ensure fire_start_date is of type datetime.date for matching purposes
        self.fire_dates['fire_start_date'] = self.fire_dates['fire_start_date'].dt.date
        # fire_dates_temp = pd.to_datetime(self.fire_dates['fire_start_date'], errors='raise')
        # self.fire_dates['fire_start_date'] = fire_dates_temp
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

        ## Initialize the variable to store the dataset
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
    def assemble_dataset(self, pipelines):
        """Assemble the dataset using the specified data pipelines"""
        logger.info(f"Wildfire Incidence Data Columns in Assembler: {self.fire_dates.columns}")
        logger.info(f"Pipeline list: {pipelines}")
        
        ## Create output folder based on the real date of the method call
        output_data_folder = f"fb_raw_output_datasets_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_data_folder, exist_ok=True)
        

        # Generate a DataFrame with all dates (fire and non-fire) for the specified period
        self.all_dates_df = self._all_dates_generator(
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.resample_interval,
            fire_dataset=self.fire_dates # Pass fire_dates that omits rows with missing fire start dates
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

        logger.debug("=====check point 1=====")
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
            logger.debug("=====check point 2=====")
            # Process each pipeline in sequence
            for pipeline in pipelines:
                # 1) CDS pipeline
                if 'EARTHKIT' in pipeline:
                    ek_pipeline = pipeline['EARTHKIT']
                    logger.info("EARTHKIT pipeline found!")

                    # Fetch weather data
                    logger.info(f"Starting request for weather data from {start_date} to {end_date}")
                    logger.debug("=====check point 3=====")
                    weather_data = ek_pipeline.ek_fetch_data(start_date, end_date)
                    logger.debug("=====check point 4=====")
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
                    logger.debug("=====check point 5=====")
                    weather_data['is_fire_day'] = weather_data.apply(self._is_fire_labeler, axis=1)
                    num_fire_days = weather_data['is_fire_day'].sum()
                    logger.info(f"Number of fire days found in this batch: {num_fire_days}")
                    logger.debug("=====check point 6=====")
                    # Store the resulting DataFrame in monthly_data
                    monthly_data = weather_data.copy()
                    logger.debug("=====check point 7=====")
                
                # 2) HUMAN_ACTIVITY pipeline
                elif 'HUMAN_ACTIVITY' in pipeline:
                    if monthly_data is None or monthly_data.empty:
                        logger.warning(f"No monthly_data from CDS to integrate with HUMAN_ACTIVITY for {period_key}.")
                        continue

                    hap_pipeline = pipeline['HUMAN_ACTIVITY']
                    logger.info("HumanActivity pipeline found!")

                    # Fetch and integrate Human Activity data
                    monthly_data = hap_pipeline.fetch_human_activity_monthly(monthly_data, period_key)
                    logger.info(f"Integrated HumanActivity data into {period_key} => final shape={monthly_data.shape}")
                    
                elif 'NED' in pipeline:
                    # NASA Earthdata pipeline assembly code here...
                    pass
            
            # After all pipelines are processed for this period, write the final CSV
            if monthly_data is not None and not monthly_data.empty:
                target_file = os.path.join(output_data_folder, f"fb_raw_data_{period_key}.csv")
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
        # NOTE: Row date is converted to pandas datetime.date object
        row['date'] = pd.to_datetime(row['date']).date()
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
    ##      - private method
    def _all_dates_generator(self, start_date, end_date, interval, fire_dataset):

        # Filter the fire dates data to only include the relevant columns and remove rows with missing values
        fire_dates = fire_dataset[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()

        # Create a DataFrame that contains every nth (interval) day from start_date to end_date
        all_dates = pd.date_range(start=start_date, end=end_date, freq=interval).normalize()

        # Convert fire_start_date to Timestamp
        fire_dates['fire_start_date'] = pd.to_datetime(fire_dates['fire_start_date']).dt.normalize()
        # fire_dates['fire_start_date'] = fire_dates['fire_start_date'].dt.normalize() 
        ## NOTE: the normalization in the line above.

        # Perform the union operation and sort the values
        all_dates = pd.Series(list(set(all_dates).union(fire_dates['fire_start_date']))).sort_values()

        # Ensure all dates are within the range start_date to end_date
        all_dates = all_dates[(all_dates >= pd.Timestamp(start_date)) & (all_dates <= pd.Timestamp(end_date))]

        # Create DataFrame for all dates without fire day labels (labeling will be done later)
        all_dates_df = pd.DataFrame({'date': all_dates})
        logger.info(f"all_dates count (constructed from fire_dates + every nth (interval) day): {len(all_dates_df)}")
        logger.debug(f"Sample all_dates:\n{all_dates_df.head()}")

        return all_dates_df