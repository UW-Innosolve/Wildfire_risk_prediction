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
            self.ab_fire_incidents = self.ab_wildfire_data_raw[required_columns].dropna()
            logger.info("Filtered wildfire incidents to required columns and dropped missing values.")
            logger.info(f"Number of wildfire (filtered) incidents loaded: {len(self.ab_fire_incidents)}")



    ## pull_additional_attr_from_raw
    ##      - add other attributes to the dataset, after the data has been temporally resampled (i.e., by wildfire_incidence_data_resample)
    def pull_additional_attr_from_raw(self):
        ## TODO: Implement this method. Additional attributes must come from the resampled data temporally and locationally.
        pass
