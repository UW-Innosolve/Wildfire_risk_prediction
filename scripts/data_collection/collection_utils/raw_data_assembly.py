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
    """
    A class to manage the raw data assembly and storage process.
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

        # Initialize the variable to store the assembled dataset
        self.dataset = None

    def assemble_dataset(self, pipelines):
        """
        Assemble the dataset using the specified data pipelines.
        - pipelines is a list of dictionaries, each with a key (e.g., 'EARTHKIT', 'HUMAN_ACTIVITY', etc.)
        - grouping_period_size is the temporal grouping period for the dataset.
        The final merged dataset is saved as CSV files in an output folder.
        """
        logger.info(f"Wildfire Incidence Data Columns in Assembler: {self.fire_dates.columns}")
        logger.info(f"Pipeline list: {pipelines}")

        # Create output folder based on the current date and time.
        output_data_folder = f"fb_raw_output_datasets_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_data_folder, exist_ok=True)

        # Generate a DataFrame with all dates (fire and non-fire) for the specified period.
        self.all_dates_df = self._all_dates_generator(
            start_date=self.start_date,
            end_date=self.end_date,
            interval=self.resample_interval,
            fire_dataset=self.fire_dates  # fire_dates that omits rows with missing fire start dates
        )

        # Group all dates by the specified period size.
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
            # Initialize temporary variables for each pipeline's output.
            monthly_data = None
            monthly_data_ek = None
            monthly_data_abltng = None
            monthly_data_hap = None

            # Start timing for this batch.
            time_start = time.time()
            logger.info(f"--- Processing batch for period: {period} ---")
            logger.info(f"Batch shape: {batch.shape}")

            start_date = batch['date'].min()
            end_date = batch['date'].max()
            period_key = period.strftime('%Y%m')

            # Process each pipeline in sequence.
            for pipeline in pipelines:
                if 'EARTHKIT' in pipeline:
                    ek_pipeline = pipeline['EARTHKIT']
                    logger.info("EARTHKIT pipeline found!")
                    logger.info(f"Starting request for earthkit data from {start_date} to {end_date}")
                    ek_data = ek_pipeline.ek_fetch_data(batch['date'])
                    if ek_data is None or ek_data.empty:
                        logger.error(f"Failed to fetch earthkit data for period {period_key}. Skipping.")
                        continue
                    if 'date' not in ek_data.columns:
                        logger.error("Weather data does not contain 'date' column. Skipping this batch.")
                        continue
                    logger.info(f"Processing weather data from {start_date} to {end_date}, Data shape: {ek_data.shape}")
                    logger.info("Labeling fire days in weather data...")
                    ek_data['is_fire_day'] = ek_data.apply(self._is_fire_labeler, axis=1)
                    num_fire_days = ek_data['is_fire_day'].sum()
                    logger.info(f"Number of fire days found in this batch: {num_fire_days}")
                    monthly_data_ek = ek_data.copy()

                elif 'HUMAN_ACTIVITY' in pipeline:
                    hap_pipeline = pipeline['HUMAN_ACTIVITY']
                    logger.info("HumanActivity pipeline found!")
                    # If no CDS data is available, create default data from the grid.
                    if monthly_data is None or monthly_data.empty:
                        logger.info(f"No CDS data available for {period_key}. Creating default data from grid.")
                        unique_dates = batch['date'].unique()
                        grid = hap_pipeline.grid  # This should have been set by set_osm_params()
                        dummy_rows = []
                        for d in unique_dates:
                            for idx, row in grid.iterrows():
                                dummy_rows.append({
                                    'date': d,
                                    'latitude': row['latitude'],
                                    'longitude': row['longitude']
                                })
                        monthly_data = pd.DataFrame(dummy_rows)
                    monthly_data_hap = hap_pipeline.fetch_human_activity_monthly(monthly_data, period_key)
                    logger.info(f"Integrated HumanActivity data into {period_key} => final shape={monthly_data_hap.shape}")

                elif 'AB_LIGHTNING' in pipeline:
                    logger.info("AB_LIGHTNING pipeline found!")
                    abltng = pipeline['AB_LIGHTNING']
                    monthly_data_abltng = abltng.get_ltng_data(batch['date']).copy()

                # Add other pipelines (e.g., NASA Earthdata) here as needed.

            pipeline_outputs_list = []

            if monthly_data_ek is not None and not monthly_data_ek.empty:
                logger.info("Earthkit data obtained, passing for assembly.")
                pipeline_outputs_list.append(monthly_data_ek)

            if monthly_data_abltng is not None and not monthly_data_abltng.empty:
                logger.info("AB Lightning data obtained, passing for assembly.")
                pipeline_outputs_list.append(monthly_data_abltng)

            if monthly_data_hap is not None and not monthly_data_hap.empty:
                logger.info("Human Activity data obtained, passing for assembly.")
                pipeline_outputs_list.append(monthly_data_hap)

            logger.info(f"Number of pipeline outputs to merge: {len(pipeline_outputs_list)}")

            # Guard against floating point errors in latitude and longitude.
            decimal_places = 2  # Adjust based on grid resolution.
            for df in pipeline_outputs_list:
                df['latitude'] = df['latitude'].round(decimal_places)
                df['longitude'] = df['longitude'].round(decimal_places)

            # Merge all pipeline outputs.
            if pipeline_outputs_list:
                monthly_data = pipeline_outputs_list[0]
                if len(pipeline_outputs_list) > 1:
                    for additional_pipeline_output in pipeline_outputs_list[1:]:
                        monthly_data = pd.merge(monthly_data,
                                                additional_pipeline_output,
                                                on=['date', 'latitude', 'longitude'], how='outer')
            else:
                monthly_data = None

            logger.info(f"Final data shape for {period_key}: {monthly_data.shape if monthly_data is not None else 'No data'}")

            # Write the final monthly CSV if data is available.
            if monthly_data is not None and not monthly_data.empty:
                target_file = os.path.join(output_data_folder, f"fb_raw_data_{period_key}.csv")
                try:
                    monthly_data.to_csv(target_file, index=False)
                    logger.info(f"Wrote monthly CSV for {period_key} -> '{target_file}'")
                except Exception as e:
                    logger.error(f"Failed to save updated weather data for {period_key}: {e}")
            else:
                logger.warning(f"No final data to save for {period_key}.")

            # End timing for this batch.
            time_end = time.time()
            logger.info(f"Processing time for this batch ({period_key}): {time_end - time_start:.2f} seconds")

    def _is_fire_labeler(self, row):
        """
        Label the fire incidents in the dataset within a specified location tolerance.
        
        Args:
            row: A single row of the DataFrame.
        
        Returns:
            int: 1 if a matching fire is found, 0 otherwise.
        """
        # Convert row date to a datetime.date object.
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

    def _all_dates_generator(self, start_date, end_date, interval, fire_dataset):
        """
        Generate a DataFrame of dates containing all fire dates from fire_dataset and
        non-fire dates resampled at the given interval.
        
        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            interval: Resample interval (e.g., '4D')
            fire_dataset: DataFrame containing fire dates.
        
        Returns:
            pd.DataFrame: DataFrame with a 'date' column of all unioned dates.
        """
        # Filter the fire dates data to include only the necessary columns.
        fire_dates = fire_dataset[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()

        # Create a DataFrame that contains every nth day from start_date to end_date.
        all_dates = pd.date_range(start=start_date, end=end_date, freq=interval).normalize()

        # Convert fire_start_date to Timestamps.
        fire_dates['fire_start_date'] = pd.to_datetime(fire_dates['fire_start_date']).dt.normalize()

        # Take the union of the generated dates and the fire start dates.
        final_unioned_dates = pd.Series(list(set(all_dates).union(fire_dates['fire_start_date']))).sort_values().dt.normalize()

        # Ensure all dates are within the specified range.
        final_unioned_dates = final_unioned_dates[
            (final_unioned_dates >= pd.Timestamp(start_date)) &
            (final_unioned_dates <= pd.Timestamp(end_date))
        ]

        # Create a DataFrame for all dates (labels will be added later).
        final_unioned_dates_df = pd.DataFrame({'date': final_unioned_dates})
        logger.info(f"final_unioned_dates_df count (constructed from fire_dates + every nth (interval) day): {len(final_unioned_dates)}")
        logger.debug(f"Sample all_dates:\n{final_unioned_dates_df.head()}")

        return final_unioned_dates_df
