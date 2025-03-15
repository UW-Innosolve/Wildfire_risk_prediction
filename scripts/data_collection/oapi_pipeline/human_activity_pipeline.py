import requests
import logging
import numpy as np
import pandas as pd

# Configure logging for the Human Activity pipeline.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class HumanActivityPipeline:
    """
    Human Activity Pipeline that retrieves infrastructure data (e.g., railways, power lines,
    highways, aerodromes, waterways) from OpenStreetMap via the Overpass API. It generates a
    spatial grid for a specified geographic area and computes counts of each feature category
    per grid cell.
    
    The output is designed to be merged with other datasets (e.g., weather data) that provide
    a 'date', 'latitude', and 'longitude' column. If no external (e.g., CDS) data is available,
    a default date–grid combination is created.
    """
    
    def __init__(self, osm_data_path=None):
        """
        Optionally provide an osm_data_path (CSV) containing pre-collected OSM data.
        If not provided, the pipeline uses Overpass API queries to retrieve features.
        """
        self.osm_data_path = osm_data_path
        self.osm_data = None
        if osm_data_path is not None:
            try:
                self.osm_data = pd.read_csv(osm_data_path)
                logger.info(f"OSM data loaded from '{osm_data_path}'.")
            except Exception as e:
                logger.error(f"Error loading OSM data from '{osm_data_path}': {e}")
                self.osm_data = None
        else:
            logger.info("No OSM data file provided; using Overpass API queries for features.")
        
        # These parameters will be set via set_osm_params.
        self.lat_range = None
        self.lon_range = None
        self.grid_resolution = None
        self.grid = None
        
        # Dictionary to hold fetched OSM features for each category.
        self.osm_features = {}
        # Define the categories to query and their corresponding Overpass filter syntax.
        self.categories = {
            "railway": '["railway"="rail"]',
            "power_line": '["power"="line"]',
            "highway": '["highway"~"^(motorway|trunk|primary|secondary)$"]',
            "aeroway": '["aeroway"="aerodrome"]',
            "waterway": '["waterway"]'
        }
    
    def set_osm_params(self, lat_range, lon_range, grid_resolution):
        """
        Set the geographic parameters for generating the spatial grid.
        
        Args:
            lat_range (list or tuple): [min_lat, max_lat]
            lon_range (list or tuple): [min_lon, max_lon]
            grid_resolution (float): resolution (in degrees) of each grid cell.
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.grid_resolution = grid_resolution
        self.grid = self._generate_grid(lat_range, lon_range, grid_resolution)
        logger.info(f"HumanActivityPipeline grid generated with {len(self.grid)} cells.")

    def _generate_grid(self, lat_range, lon_range, grid_resolution):
        """
        Generate a spatial grid (DataFrame) covering the specified area.
        
        Returns:
            pd.DataFrame: DataFrame with columns 'latitude' and 'longitude' for grid cell corners.
        """
        # Ensure the range covers the endpoint by adding grid_resolution.
        lat_values = np.arange(lat_range[0], lat_range[1] + grid_resolution, grid_resolution)
        lon_values = np.arange(lon_range[0], lon_range[1] + grid_resolution, grid_resolution)
        grid = pd.DataFrame(
            [(lat, lon) for lat in lat_values for lon in lon_values],
            columns=['latitude', 'longitude']
        )
        return grid

    def _simulate_human_activity(self, lat, lon):
        """
        Simulate a human activity index based on distance from the center of the area.
        (This function is a placeholder and can be replaced with real data queries.)
        """
        center_lat = (self.lat_range[0] + self.lat_range[1]) / 2
        center_lon = (self.lon_range[0] + self.lon_range[1]) / 2
        distance = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
        max_distance = np.sqrt(((self.lat_range[1] - self.lat_range[0]) / 2) ** 2 +
                               ((self.lon_range[1] - self.lon_range[0]) / 2) ** 2)
        activity = 100 * (1 - distance / max_distance)
        return max(activity, 0)

    def fetch_infrastructure_data(self):
        """
        Fetch infrastructure features from OpenStreetMap using the Overpass API for each defined category.
        Fetched data are stored in self.osm_features as DataFrames with columns 'lat' and 'lon'.
        """
        if self.lat_range is None or self.lon_range is None:
            raise ValueError("OSM parameters not set. Call set_osm_params before fetching data.")
            
        # Define the bounding box in the format: south,west,north,east.
        bbox = f"{self.lat_range[0]},{self.lon_range[0]},{self.lat_range[1]},{self.lon_range[1]}"
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        for category, osm_filter in self.categories.items():
            query = f"""
            [out:json];
            (
              node{osm_filter}({bbox});
              way{osm_filter}({bbox});
              relation{osm_filter}({bbox});
            );
            out center;
            """
            try:
                logger.info(f"Querying Overpass for category '{category}' with filter {osm_filter}")
                response = requests.get(overpass_url, params={'data': query}, timeout=60)
                data = response.json()
                elements = data.get('elements', [])
                coords = []
                for element in elements:
                    if 'lat' in element and 'lon' in element:
                        coords.append((element['lat'], element['lon']))
                    elif 'center' in element:
                        coords.append((element['center']['lat'], element['center']['lon']))
                df = pd.DataFrame(coords, columns=['lat', 'lon'])
                logger.info(f"Fetched {len(df)} features for category '{category}'.")
                self.osm_features[category] = df
            except Exception as e:
                logger.error(f"Error fetching OSM data for category '{category}': {e}")
                self.osm_features[category] = pd.DataFrame(columns=['lat', 'lon'])

    def _compute_infrastructure_counts(self):
        """
        For each grid cell in the generated grid, compute the count of OSM features for each category.
        
        Returns:
            pd.DataFrame: The grid DataFrame augmented with count columns (e.g., 'railway_count').
        """
        infrastructure_counts = self.grid.copy()
        for category, features_df in self.osm_features.items():
            counts = []
            for idx, row in self.grid.iterrows():
                count = features_df[
                    (features_df['lat'] >= row['latitude']) &
                    (features_df['lat'] < row['latitude'] + self.grid_resolution) &
                    (features_df['lon'] >= row['longitude']) &
                    (features_df['lon'] < row['longitude'] + self.grid_resolution)
                ].shape[0]
                counts.append(count)
            infrastructure_counts[f"{category}_count"] = counts
        return infrastructure_counts

    def fetch_human_activity_monthly(self, monthly_data, period_key):
        """
        Integrate OSM infrastructure data into the monthly parameters.
        
        Steps:
          1. If infrastructure data hasn't been fetched, query it via the Overpass API.
          2. Compute counts of features per grid cell.
          3. Create a DataFrame for all combinations of the unique dates in monthly_data and grid cells.
          4. Merge the infrastructure counts with this date grid.
          5. Finally, merge the result with monthly_data.
          
        Args:
            monthly_data (pd.DataFrame): DataFrame that must include 'date', 'latitude', and 'longitude' columns.
            period_key (str): Identifier for the current period (e.g., "201401").
            
        Returns:
            pd.DataFrame: Merged DataFrame with additional infrastructure count columns.
        """
        if self.grid is None:
            raise ValueError("OSM parameters not set. Please call set_osm_params before fetching data.")

        # Fetch OSM infrastructure data if not already done.
        if not self.osm_features:
            self.fetch_infrastructure_data()

        # Compute counts per grid cell.
        infra_counts = self._compute_infrastructure_counts()

        # Round coordinates in both datasets to avoid floating point mismatches.
        monthly_data['latitude'] = monthly_data['latitude'].round(2)
        monthly_data['longitude'] = monthly_data['longitude'].round(2)
        infra_counts['latitude'] = infra_counts['latitude'].round(2)
        infra_counts['longitude'] = infra_counts['longitude'].round(2)

        # Create a DataFrame of all combinations of unique dates and grid cells.
        unique_dates = monthly_data['date'].unique()
        date_grid = pd.DataFrame([
            (d, row['latitude'], row['longitude']) 
            for d in unique_dates 
            for idx, row in infra_counts.iterrows()
        ], columns=['date', 'latitude', 'longitude'])

        # Merge the infrastructure counts with the date grid.
        merged_infra = pd.merge(date_grid, infra_counts, on=['latitude', 'longitude'], how='left')
        # Merge with the original monthly_data.
        merged = pd.merge(monthly_data, merged_infra, on=['date', 'latitude', 'longitude'], how='outer')
        logger.info(f"Infrastructure data integrated: merged shape is {merged.shape}.")
        return merged
