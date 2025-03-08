import logging
import os
import pandas as pd
from oapi_pipeline.human_activity_pipeline import HumanActivityPipeline

# Configure logging: output both to file and console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    handlers=[
        logging.FileHandler("test_activity.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_human_activity_pipeline():
    """
    A standalone test script to test the HumanActivityPipeline in isolation.
    This script:
      1. Initializes the HumanActivityPipeline.
      2. Sets the OSM parameters (latitude range, longitude range, grid resolution).
      3. Creates a dummy DataFrame with a sample date and a few sample latitude/longitude points.
      4. Calls fetch_human_activity_monthly() to integrate infrastructure data.
      5. Logs the input and output DataFrames.
    """
    
    # Initialize the pipeline
    pipeline = HumanActivityPipeline()
    
    # Set the OSM parameters: adjust these values as needed.
    # Here we use a small area around the Genesee Power Plant in Alberta.
    pipeline.set_osm_params(
        lat_range=[53.32, 53.35],
        lon_range=[-114.33, -114.29],
        grid_resolution=0.01
    )
    
    # Create a dummy DataFrame for testing:
    # We'll use a single sample date and three sample points.
    sample_date = pd.to_datetime("2025-03-04")
    sample_points = [
        (53.331, -114.32),  # Genesee plant area
        (53.340, -114.30),  # Slightly north/east
        (53.335, -114.31),  # Another close spot
    ]
    # Build the DataFrame with columns: 'date', 'latitude', 'longitude'
    df = pd.DataFrame([
        {"date": sample_date, "latitude": lat, "longitude": lon}
        for (lat, lon) in sample_points
    ])
    
    logger.info("Input DataFrame:")
    logger.info(df)
    
    # Call fetch_human_activity_monthly to integrate infrastructure data.
    # The method expects a DataFrame with a 'date' column and spatial coordinates,
    # and a period key (here, we use "TEST" for testing purposes).
    result_df = pipeline.fetch_human_activity_monthly(df, "TEST")
    
    logger.info("Output DataFrame:")
    logger.info(result_df)

if __name__ == "__main__":
    test_human_activity_pipeline()
