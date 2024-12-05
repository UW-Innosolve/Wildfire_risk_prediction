import pandas as pd
from datetime import timedelta

class AlbertaWildfireIncidenceLoader:
    """A class to manage the Alberta wildfire incidence data loading process"""

    def __init__(self, wildfire_data_path):
        self.wildfire_data_path = wildfire_data_path
        # Load wildfire data (ie. wildfire incidence data)
        self.ab_wildfire_data_raw = pd.read_excel("scripts/data_collection/fp-historical-wildfire-data-2006-2023.xlsx")
        # Convert 'fire_start_date' to datetime format and extract only the date part
        self.ab_wildfire_data_raw['fire_start_date'] = pd.to_datetime(self.ab_wildfire_data_raw['fire_start_date'], errors='coerce')
        # Filter the fire dates data to only include the fire start date and location, also remove rows with missing values
        self.ab_fire_incidents = self.ab_wildfire_data_raw[['fire_start_date', 'fire_location_latitude', 'fire_location_longitude']].dropna()


    ## wildfire_incidence_data_resample
    ##      - resample the wildfire incidence data to include EVERY FIRE DAY with the NON-FIRE DAYS resampled to the specified interval
    ##      - input: start_date, end_date, interval (eg. '4D' is every 4th day), fire_incident_data
    ##      - output: fire_incident_data_resampled (pandas DataFrame)
    ##      - (NOTE: fire_incidnet_data would typically be self.ab_fire_incidents which was set at class initialization)
    def wildfire_incidence_data_resample(self, start_date, end_date, interval, fire_incident_data):
        """Resample the wildfire incidence data to include every fire day with the non-fire days resampled to the specified interval"""
        resampled_date_range = pd.date_range(start=start_date, end=end_date, freq=interval).normalize()
        fire_incident_data_resampled = pd.Series(list(set(resampled_date_range).union(fire_incident_data['fire_start_date']))).sort_values()
        fire_incident_data_resampled = fire_incident_data_resampled[(fire_incident_data_resampled >= pd.Timestamp(start_date)) & (fire_incident_data_resampled <= pd.Timestamp(end_date))] ## Ensure all dates are within the range 2006-2023 (NOTE: redundant?)
        fire_incident_data_resampled = pd.DataFrame({'date': fire_incident_data_resampled})
        return fire_incident_data_resampled


    ## pull_additional_attr_from_raw
    ##      - add other attributes to the dataset, after the data has been temporally resampled (ie. by wildfire_incidence_data_resample)
    def pull_additional_attr_from_raw(self):
        ## TODO: This impliment method, additional attributes must come from the with resampled data temporally and locationally
        pass



