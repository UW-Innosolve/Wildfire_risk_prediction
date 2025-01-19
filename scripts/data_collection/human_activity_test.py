# humanactivitytest.py

import logging
import os

from human_activity_pipeline import HumanActivityPipeline

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
    A standalone test script that mimics minimal usage of the
    HumanActivityPipeline without the rest of the pipeline logic.
    """

    # Ensure the pipeline is initialized
    pipeline = HumanActivityPipeline()

    # We'll pick a few sample lat/lon points around Genesee Power Plant in Alberta
    # known for industrial/power lines.
    sample_points = [
        (53.331, -114.32),  # Genesee plant area
        (53.340, -114.30),  # Slightly north/east
        (53.335, -114.31),  # Another close spot
    ]

    results = []
    for (lat, lon) in sample_points:
        logger.info(f"Fetching human activity data for lat={lat}, lon={lon}")
        ha_df = pipeline.fetch_human_activity(lat, lon)
        logger.info(f"Received:\n{ha_df}")
        results.append(ha_df)

    # Combine all results into a single DataFrame for demonstration
    combined = None
    if results:
        # Start with the first DataFrame
        combined = results[0]
        if len(results) > 1:
            # Use pd.concat instead of append (pandas 1.4+)
            import pandas as pd
            combined = pd.concat([combined] + results[1:], ignore_index=True)

    logger.info("=== Final Combined Results ===")
    logger.info(f"\n{combined}")

if __name__ == "__main__":
    test_human_activity_pipeline()
