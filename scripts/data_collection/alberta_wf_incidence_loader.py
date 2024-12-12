import pandas as pd
from datetime import timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlbertaWildfireIncidenceLoader:
    """A class to manage the Alberta wildfire incidence data loading process"""

    def __init__(self, wildfire_data_path):
        self.wildfire_data_path = wildfire_data_path
        # Load wildfire data (i.e., wildfire incidence data)
        try:
            self.ab_wildfire_data_raw = pd.read_excel(self.wildfire_data_path)
            logger.info(f"Wildfire data loaded from '{self.wildfire_data_path}'.")
        except Exception as e:
            logger.error(f"Failed to load wildfire data from '{self.wildfire_data_path}': {e}")
            self.ab_wildfire_data_raw = pd.DataFrame()

        # Convert 'fire_start_date' to datetime format and extract only the date part
        if 'fire_start_date' in self.ab_wildfire_data_raw.columns:
            self.ab_wildfire_data_raw['fire_start_date'] = pd.to_datetime(
                self.ab_wildfire_data_raw['fire_start_date'], errors='coerce'
            )
            logger.info("Converted 'fire_start_date' to datetime format.")
        else:
            logger.error("'fire_start_date' column not found in wildfire data.")
            self.ab_wildfire_data_raw['fire_start_date'] = pd.NaT

        # Filter the fire dates data to only include the fire start date and location, also remove rows with missing values
        required_columns = ['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']
        missing_columns = [col for col in required_columns if col not in self.ab_wildfire_data_raw.columns]
        if missing_columns:
            logger.error(f"Missing required columns in wildfire data: {missing_columns}")
            self.ab_fire_incidents = pd.DataFrame(columns=required_columns)
        else:
            self.ab_fire_incidents = self.ab_wildfire_data_raw[
                required_columns
            ].dropna()
            logger.info("Filtered wildfire incidents to required columns and dropped missing values.")
            logger.info(f"Number of wildfire incidents loaded: {len(self.ab_fire_incidents)}")

    ## wildfire_incidence_data_resample
    ##      - resample the wildfire incidence data to include EVERY FIRE DAY with the NON-FIRE DAYS resampled to the specified interval
    ##      - input: start_date, end_date, interval (e.g., '4D' is every 4th day), fire_incident_data
    ##      - output: fire_incident_data_resampled (pandas DataFrame)
    ##      - (NOTE: fire_incident_data would typically be self.ab_fire_incidents which was set at class initialization)
    def wildfire_incidence_data_resample(self, start_date, end_date, interval, fire_incident_data):
        """Resample the wildfire incidence data to include every fire day with the non-fire days resampled to the specified interval"""
        try:
            # Create a DataFrame with resampled dates
            resampled_date_range = pd.date_range(start=start_date, end=end_date, freq=interval).normalize()
            resampled_dates_df = pd.DataFrame({'fire_start_date': resampled_date_range})
            
            # Merge with fire_incident_data to get location data for fire dates
            fire_dates_df = fire_incident_data[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']]
            resampled_fire_incidents = resampled_dates_df.merge(fire_dates_df, on='fire_start_date', how='left')
            
            # Ensure all dates are within the specified range
            resampled_fire_incidents = resampled_fire_incidents[
                (resampled_fire_incidents['fire_start_date'] >= pd.Timestamp(start_date)) &
                (resampled_fire_incidents['fire_start_date'] <= pd.Timestamp(end_date))
            ]
            
            logger.info("Resampled wildfire incidence data successfully.")
            logger.info(f"Total resampled dates: {len(resampled_fire_incidents)}")
            return resampled_fire_incidents
        except Exception as e:
            logger.error(f"Error during resampling wildfire incidence data: {e}")
            return pd.DataFrame(columns=['fire_start_date', 'fire_location_latitude', 'fire_location_longitude'])

    ## pull_additional_attr_from_raw
    ##      - add other attributes to the dataset, after the data has been temporally resampled (i.e., by wildfire_incidence_data_resample)
    def pull_additional_attr_from_raw(self):
        ## TODO: Implement this method. Additional attributes must come from the resampled data temporally and locationally.
        pass
